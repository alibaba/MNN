// sherpa-mnn/csrc/parse-options.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation
//
// This file is copied and modified from kaldi/src/util/parse-options.h

#ifndef SHERPA_ONNX_CSRC_PARSE_OPTIONS_H_
#define SHERPA_ONNX_CSRC_PARSE_OPTIONS_H_

#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sherpa_mnn {

class ParseOptions {
 public:
  explicit ParseOptions(const char *usage)
      : print_args_(true),
        help_(false),
        usage_(usage),
        argc_(0),
        argv_(nullptr),
        prefix_(""),
        other_parser_(nullptr) {
#if !defined(_MSC_VER) && !defined(__CYGWIN__)
    // This is just a convenient place to set the stderr to line
    // buffering mode, since it's called at program start.
    // This helps ensure different programs' output is not mixed up.
    setlinebuf(stderr);
#endif
    RegisterStandard("config", &config_,
                     "Configuration file to read (this "
                     "option may be repeated)");
    RegisterStandard("print-args", &print_args_,
                     "Print the command line arguments (to stderr)");
    RegisterStandard("help", &help_, "Print out usage message");
  }

  /**
    This is a constructor for the special case where some options are
    registered with a prefix to avoid conflicts.  The object thus created will
    only be used temporarily to register an options class with the original
    options parser (which is passed as the *other pointer) using the given
    prefix.  It should not be used for any other purpose, and the prefix must
    not be the empty string.  It seems to be the least bad way of implementing
    options with prefixes at this point.
    Example of usage is:
     ParseOptions po;  // original ParseOptions object
     ParseOptions po_mfcc("mfcc", &po); // object with prefix.
     MfccOptions mfcc_opts;
     mfcc_opts.Register(&po_mfcc);
    The options will now get registered as, e.g., --mfcc.frame-shift=10.0
    instead of just --frame-shift=10.0
   */
  ParseOptions(const std::string &prefix, ParseOptions *other);

  ParseOptions(const ParseOptions &) = delete;
  ParseOptions &operator=(const ParseOptions &) = delete;
  ~ParseOptions() = default;

  void Register(const std::string &name, bool *ptr, const std::string &doc);
  void Register(const std::string &name, int32_t *ptr, const std::string &doc);
  void Register(const std::string &name, int64_t *ptr, const std::string &doc);
  void Register(const std::string &name, uint32_t *ptr, const std::string &doc);
  void Register(const std::string &name, float *ptr, const std::string &doc);
  void Register(const std::string &name, double *ptr, const std::string &doc);
  void Register(const std::string &name, std::string *ptr,
                const std::string &doc);

  /// If called after registering an option and before calling
  /// Read(), disables that option from being used.  Will crash
  /// at runtime if that option had not been registered.
  void DisableOption(const std::string &name);

  /// This one is used for registering standard parameters of all the programs
  template <typename T>
  void RegisterStandard(const std::string &name, T *ptr,
                        const std::string &doc);

  /**
    Parses the command line options and fills the ParseOptions-registered
    variables. This must be called after all the variables were registered!!!

    Initially the variables have implicit values,
    then the config file values are set-up,
    finally the command line values given.
    Returns the first position in argv that was not used.
    [typically not useful: use NumParams() and GetParam(). ]
   */
  int Read(int argc, const char *const *argv);

  /// Prints the usage documentation [provided in the constructor].
  void PrintUsage(bool print_command_line = false) const;

  /// Prints the actual configuration of all the registered variables
  void PrintConfig(std::ostream &os) const;

  /// Reads the options values from a config file.  Must be called after
  /// registering all options.  This is usually used internally after the
  /// standard --config option is used, but it may also be called from a
  /// program.
  void ReadConfigFile(const std::string &filename);

  /// Number of positional parameters (c.f. argc-1).
  int NumArgs() const;

  /// Returns one of the positional parameters; 1-based indexing for argc/argv
  /// compatibility. Will crash if param is not >=1 and <=NumArgs().
  ///
  /// Note: Index is 1 based.
  std::string GetArg(int param) const;

  std::string GetOptArg(int param) const {
    return (param <= NumArgs() ? GetArg(param) : "");
  }

  /// The following function will return a possibly quoted and escaped
  /// version of "str", according to the current shell.  Currently
  /// this is just hardwired to bash.  It's useful for debug output.
  static std::string Escape(const std::string &str);

 private:
  /// Template to register various variable types,
  /// used for program-specific parameters
  template <typename T>
  void RegisterTmpl(const std::string &name, T *ptr, const std::string &doc);

  // Following functions do just the datatype-specific part of the job
  /// Register boolean variable
  void RegisterSpecific(const std::string &name, const std::string &idx,
                        bool *b, const std::string &doc, bool is_standard);
  /// Register int32_t variable
  void RegisterSpecific(const std::string &name, const std::string &idx,
                        int32_t *i, const std::string &doc, bool is_standard);
  /// Register int64_t variable
  void RegisterSpecific(const std::string &name, const std::string &idx,
                        int64_t *i, const std::string &doc, bool is_standard);
  /// Register unsigned  int32_t variable
  void RegisterSpecific(const std::string &name, const std::string &idx,
                        uint32_t *u, const std::string &doc, bool is_standard);
  /// Register float variable
  void RegisterSpecific(const std::string &name, const std::string &idx,
                        float *f, const std::string &doc, bool is_standard);
  /// Register double variable [useful as we change BaseFloat type].
  void RegisterSpecific(const std::string &name, const std::string &idx,
                        double *f, const std::string &doc, bool is_standard);
  /// Register string variable
  void RegisterSpecific(const std::string &name, const std::string &idx,
                        std::string *s, const std::string &doc,
                        bool is_standard);

  /// Does the actual job for both kinds of parameters
  /// Does the common part of the job for all datatypes,
  /// then calls RegisterSpecific
  template <typename T>
  void RegisterCommon(const std::string &name, T *ptr, const std::string &doc,
                      bool is_standard);

  /// Set option with name "key" to "value"; will crash if can't do it.
  /// "has_equal_sign" is used to allow --x for a boolean option x,
  /// and --y=, for a string option y.
  bool SetOption(const std::string &key, const std::string &value,
                 bool has_equal_sign);

  bool ToBool(std::string str) const;
  int32_t ToInt(const std::string &str) const;
  int64_t ToInt64(const std::string &str) const;
  uint32_t ToUint(const std::string &str) const;
  float ToFloat(const std::string &str) const;
  double ToDouble(const std::string &str) const;

  // maps for option variables
  std::unordered_map<std::string, bool *> bool_map_;
  std::unordered_map<std::string, int32_t *> int_map_;
  std::unordered_map<std::string, int64_t *> int64_map_;
  std::unordered_map<std::string, uint32_t *> uint_map_;
  std::unordered_map<std::string, float *> float_map_;
  std::unordered_map<std::string, double *> double_map_;
  std::unordered_map<std::string, std::string *> string_map_;

  /**
     Structure for options' documentation
   */
  struct DocInfo {
    DocInfo() = default;
    DocInfo(const std::string &name, const std::string &usemsg)
        : name_(name), use_msg_(usemsg), is_standard_(false) {}
    DocInfo(const std::string &name, const std::string &usemsg,
            bool is_standard)
        : name_(name), use_msg_(usemsg), is_standard_(is_standard) {}

    std::string name_;
    std::string use_msg_;
    bool is_standard_;
  };
  using DocMapType = std::unordered_map<std::string, DocInfo>;
  DocMapType doc_map_;  ///< map for the documentation

  bool print_args_;     ///< variable for the implicit --print-args parameter
  bool help_;           ///< variable for the implicit --help parameter
  std::string config_;  ///< variable for the implicit --config parameter
  std::vector<std::string> positional_args_;
  const char *usage_;
  int argc_;
  const char *const *argv_;

  /// These members are not normally used. They are only used when the object
  /// is constructed with a prefix
  std::string prefix_;
  ParseOptions *other_parser_;

 protected:
  /// SplitLongArg parses an argument of the form --a=b, --a=, or --a,
  /// and sets "has_equal_sign" to true if an equals-sign was parsed..
  /// this is needed in order to correctly allow --x for a boolean option
  /// x, and --y= for a string option y, and to disallow --x= and --y.
  void SplitLongArg(const std::string &in, std::string *key, std::string *value,
                    bool *has_equal_sign) const;

  void NormalizeArgName(std::string *str) const;

  /// Removes the beginning and trailing whitespaces from a string
  void Trim(std::string *str) const;
};

/// This template is provided for convenience in reading config classes from
/// files; this is not the standard way to read configuration options, but may
/// occasionally be needed.  This function assumes the config has a function
/// "void Register(ParseOptions *opts)" which it can call to register the
/// ParseOptions object.
template <class C>
void ReadConfigFromFile(const std::string &config_filename, C *c) {
  std::ostringstream usage_str;
  usage_str << "Parsing config from "
            << "from '" << config_filename << "'";
  ParseOptions po(usage_str.str().c_str());
  c->Register(&po);
  po.ReadConfigFile(config_filename);
}

/// This variant of the template ReadConfigFromFile is for if you need to read
/// two config classes from the same file.
template <class C1, class C2>
void ReadConfigsFromFile(const std::string &conf, C1 *c1, C2 *c2) {
  std::ostringstream usage_str;
  usage_str << "Parsing config from "
            << "from '" << conf << "'";
  ParseOptions po(usage_str.str().c_str());
  c1->Register(&po);
  c2->Register(&po);
  po.ReadConfigFile(conf);
}

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_PARSE_OPTIONS_H_
