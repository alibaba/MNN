#include <map>
#include <string>
#include <vector>

#include "piper/phoneme_ids.hpp"

void phonemes_to_ids(const std::vector<Phoneme> &phonemes, PhonemeIdConfig &config,
                     std::vector<PhonemeId> &phonemeIds,
                     std::map<Phoneme, std::size_t> &missingPhonemes)
{

  auto phonemeIdMap = std::make_shared<PhonemeIdMap>(DEFAULT_PHONEME_ID_MAP);
  if (config.phonemeIdMap)
  {
    phonemeIdMap = config.phonemeIdMap;
  }

  // Beginning of sentence symbol (^)
  if (config.addBos)
  {
    auto const bosIds = &(phonemeIdMap->at(config.bos));
    phonemeIds.insert(phonemeIds.end(), bosIds->begin(), bosIds->end());

    if (config.interspersePad)
    {
      // Pad after bos (_)
      auto const padIds = &(phonemeIdMap->at(config.pad));
      phonemeIds.insert(phonemeIds.end(), padIds->begin(), padIds->end());
    }
  }

  if (config.interspersePad)
  {
    // Add ids for each phoneme *with* padding
    auto const padIds = &(phonemeIdMap->at(config.pad));

    for (auto const phoneme : phonemes)
    {
      if (phonemeIdMap->count(phoneme) < 1)
      {
        // Phoneme is missing from id map
        if (missingPhonemes.count(phoneme) < 1)
        {
          missingPhonemes[phoneme] = 1;
        }
        else
        {
          missingPhonemes[phoneme] += 1;
        }

        continue;
      }

      auto const mappedIds = &(phonemeIdMap->at(phoneme));
      phonemeIds.insert(phonemeIds.end(), mappedIds->begin(), mappedIds->end());

      // pad (_)
      phonemeIds.insert(phonemeIds.end(), padIds->begin(), padIds->end());
    }
  }
  else
  {
    // Add ids for each phoneme *without* padding
    for (auto const phoneme : phonemes)
    {
      auto const mappedIds = &(phonemeIdMap->at(phoneme));
      phonemeIds.insert(phonemeIds.end(), mappedIds->begin(), mappedIds->end());
    }
  }

  // End of sentence symbol ($)
  if (config.addEos)
  {
    auto const eosIds = &(phonemeIdMap->at(config.eos));
    phonemeIds.insert(phonemeIds.end(), eosIds->begin(), eosIds->end());
  }
}
