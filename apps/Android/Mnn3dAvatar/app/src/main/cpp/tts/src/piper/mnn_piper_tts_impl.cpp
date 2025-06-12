
#include "piper/mnn_piper_tts_impl.hpp"
#include "piper/utf8.h"
#include <mutex>
#include <codecvt> // For std::wstring_convert and std::codecvt_utf8
#include <locale>

MNNPiperTTSImpl::MNNPiperTTSImpl(const std::string &espeak_data_path, const std::string &model_path, const std::string &cache_path)
{
  audio_generator_ = AudioGenerator(model_path);
  phone_id_map_ = DEFAULT_PHONEME_ID_MAP;

  // Initialize espeak-ng
  // The 'path_data' argument is crucial. It should point to the 'espeak-ng-data' directory.
  // Assuming espeak-ng is built within the project and data is available relative to executable or a known path.
  // For this example, let's assume 'espeak-ng-data' is in the same directory as the executable or a path needs to be configured.
  // For a proper setup, this path might need to be determined dynamically or configured.
  char path_data[1024];
  PLOG(INFO, "espeak data path: " + espeak_data_path);
  strncpy(path_data, espeak_data_path.c_str(), sizeof(path_data) - 1);
  path_data[sizeof(path_data) - 1] = 0;

  espeak_AUDIO_OUTPUT output_type = AUDIO_OUTPUT_SYNCHRONOUS;
  int buflength = 500;
  int options = 0; // espeakINITIALIZE_PHONEME_EVENTS; // Example option, adjust as needed

  if (espeak_Initialize(output_type, buflength, path_data, options) == EE_INTERNAL_ERROR)
  {
    std::runtime_error("Failed to initialize espeak-ng. Ensure espeak-ng-data path is correct.");
  }

  int result = espeak_SetVoiceByName("en-us");
  if (result != 0)
  {
    throw std::runtime_error("Failed to set eSpeak-ng voice");
  }
}
void MNNPiperTTSImpl::phonemize_eSpeak(std::string text, std::vector<std::vector<Phoneme>> &phonemes)
{
  // Modified by eSpeak
  std::string textCopy(text);

  std::vector<Phoneme> *sentencePhonemes = nullptr;
  const char *inputTextPointer = textCopy.c_str();
  int terminator = 0;

  while (inputTextPointer != NULL)
  {
    // Modified espeak-ng API to get access to clause terminator
    std::string clausePhonemes(espeak_TextToPhonemesWithTerminator(
        (const void **)&inputTextPointer,
        /*textmode*/ espeakCHARS_AUTO,
        /*phonememode = IPA*/ 0x02, &terminator));

    // Decompose, e.g. "ç" -> "c" + "̧"
    auto phonemesNorm = una::norm::to_nfd_utf8(clausePhonemes);
    auto phonemesRange = una::ranges::utf8_view{phonemesNorm};

    if (!sentencePhonemes)
    {
      // Start new sentence
      phonemes.emplace_back();
      sentencePhonemes = &phonemes[phonemes.size() - 1];
    }

    // Maybe use phoneme map
    std::vector<Phoneme> mappedSentPhonemes;

    mappedSentPhonemes.insert(mappedSentPhonemes.end(), phonemesRange.begin(),
                              phonemesRange.end());

    auto phonemeIter = mappedSentPhonemes.begin();
    auto phonemeEnd = mappedSentPhonemes.end();

    // Filter out (lang) switch (flags).
    // These surround words from languages other than the current voice.
    bool inLanguageFlag = false;

    while (phonemeIter != phonemeEnd)
    {
      if (inLanguageFlag)
      {
        if (*phonemeIter == U')')
        {
          // End of (lang) switch
          inLanguageFlag = false;
        }
      }
      else if (*phonemeIter == U'(')
      {
        // Start of (lang) switch
        inLanguageFlag = true;
      }
      else
      {
        sentencePhonemes->push_back(*phonemeIter);
      }

      phonemeIter++;
    }

    // Add appropriate punctuation depending on terminator type
    int punctuation = terminator & 0x000FFFFF;
    if (punctuation == CLAUSE_PERIOD)
    {
      sentencePhonemes->push_back(period);
    }
    else if (punctuation == CLAUSE_QUESTION)
    {
      sentencePhonemes->push_back(question);
    }
    else if (punctuation == CLAUSE_EXCLAMATION)
    {
      sentencePhonemes->push_back(exclamation);
    }
    else if (punctuation == CLAUSE_COMMA)
    {
      sentencePhonemes->push_back(comma);
      sentencePhonemes->push_back(space);
    }
    else if (punctuation == CLAUSE_COLON)
    {
      sentencePhonemes->push_back(colon);
      sentencePhonemes->push_back(space);
    }
    else if (punctuation == CLAUSE_SEMICOLON)
    {
      sentencePhonemes->push_back(semicolon);
      sentencePhonemes->push_back(space);
    }

    if ((terminator & CLAUSE_TYPE_SENTENCE) == CLAUSE_TYPE_SENTENCE)
    {
      // End of sentence
      sentencePhonemes = nullptr;
    }
  }
}

std::vector<int16_t> MNNPiperTTSImpl::synthesize(std::vector<PhonemeId> &phonemeIds)
{

  std::vector<int> input;
  for (int i = 0; i < phonemeIds.size(); i++)
  {
    input.push_back(phonemeIds[i]);
  }
  std::vector<float> scales{0.667, 1.0, 0.8};
  auto audio = audio_generator_.Process(input, input.size(), scales);

  std::vector<int16_t> audioBuffer;
  for (int i = 0; i < audio.size(); i++)
  {
    audioBuffer.push_back(audio[i] * 32768);
  }
  return audioBuffer;
}

std::tuple<int, Audio> MNNPiperTTSImpl::Process(const std::string& text)
{
  std::lock_guard<std::mutex> lock(mtx_); // Lock the mutex

  auto t0 = clk::now();

  std::size_t sentenceSilenceSamples = 0;

  // Phonemes for each sentence
  std::vector<std::vector<Phoneme>> phonemes;

  CodepointsPhonemeConfig codepointsConfig;
  phonemize_eSpeak(text, phonemes);

  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
  // Synthesize each sentence independently.
  std::vector<PhonemeId> phonemeIds;
  std::map<Phoneme, std::size_t> missingPhonemes;
  std::vector<int16_t> audioBufferFinal;
  for (auto phonemesIter = phonemes.begin(); phonemesIter != phonemes.end();
       ++phonemesIter)
  {
    std::vector<Phoneme> &sentencePhonemes = *phonemesIter;

    std::vector<std::shared_ptr<std::vector<Phoneme>>> phrasePhonemes;

    // Use phoneme/id map from config
    PhonemeIdConfig idConfig;
    idConfig.phonemeIdMap = std::make_shared<PhonemeIdMap>(phone_id_map_);

    // Use all phonemes
    phrasePhonemes.push_back(
        std::make_shared<std::vector<Phoneme>>(sentencePhonemes));

    // phonemes -> ids -> audio
    for (size_t phraseIdx = 0; phraseIdx < phrasePhonemes.size(); phraseIdx++)
    {
      if (phrasePhonemes[phraseIdx]->size() <= 0)
      {
        continue;
      }
      auto utf32_vector = phrasePhonemes[phraseIdx];
      std::string utf8_string = converter.to_bytes(utf32_vector->data(), utf32_vector->data() + utf32_vector->size());
      PLOG(INFO, "TTS phonemes: " + utf8_string);

      // phonemes -> ids
      phonemes_to_ids(*(phrasePhonemes[phraseIdx]), idConfig, phonemeIds,
                      missingPhonemes);

      // ids -> audio
      auto audioBuffer = synthesize(phonemeIds);
      audioBufferFinal.insert(audioBufferFinal.end(), audioBuffer.begin(),
                              audioBuffer.end());
      phonemeIds.clear();
    }

    phonemeIds.clear();
  }

  auto t1 = clk::now();
  auto duration_total = std::chrono::duration_cast<ms>(t1 - t0);
  float timecost_in_ms = (float)duration_total.count();
  float audio_len_in_ms = (float)audioBufferFinal.size() / (float)sample_rate_ * (float)1000;
  float rtf = timecost_in_ms / audio_len_in_ms;

  PLOG(INFO, "TTS timecost: " + std::to_string(timecost_in_ms) + "ms, audio_duration: " + std::to_string(audio_len_in_ms) + "ms, rtf:" + std::to_string(rtf));

  return std::make_tuple(sample_rate_, audioBufferFinal);
}