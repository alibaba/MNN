//
//  mnncli.cpp
//
//  Created by MNN on 2023/03/24.
//  Jinde.Song
//  LLM command line tool, based on llm_demo.cpp
//
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include "cli_command_handler.hpp"
#include "cli_command_parser.hpp"
#include "cli_command_spec.hpp"
#include "handlers/run_command_handler.hpp"
#include "handlers/delete_command_handler.hpp"
#include "handlers/list_command_handler.hpp"
#include "handlers/search_command_handler.hpp"
#include "handlers/download_command_handler.hpp"
#include "handlers/model_info_command_handler.hpp"
#include "handlers/serve_command_handler.hpp"
#include "handlers/benchmark_command_handler.hpp"
#include "handlers/config_command_handler.hpp"
#include "handlers/info_command_handler.hpp"
#include "log_utils.hpp"
#include "user_interface.hpp"


// Forward declarations
class CommandLineInterface;

// Performance evaluation using ModelRunner

// Main command line interface
class CommandLineInterface {
public:
    CommandLineInterface() : verbose_(false) {}
    
    int Run(int argc, const char* argv[]) {
        if (argc < 2) {
            PrintUsage();
            return 0;
        }
        
        std::string cmd_name = (argc >= 2) ? argv[1] : "";
        
        // List of supported handler-based commands
        static const std::vector<std::string> handler_commands = {
            "run", "delete", "list", "search", "download", "model_info", "serve", "benchmark", "config", "info"
        };
        
        // Check if this is a handler-based command
        bool is_handler_command = std::find(handler_commands.begin(), handler_commands.end(), cmd_name) != handler_commands.end();
        
        if (is_handler_command) {
            try {
                // Initialize dispatcher and parser
                mnncli::CommandDispatcher dispatcher;
                mnncli::CommandParser parser(argc, argv);
                
                // Register all handlers
                dispatcher.Register(std::make_unique<mnncli::RunCommandHandler>());
                dispatcher.Register(std::make_unique<mnncli::DeleteCommandHandler>());
                dispatcher.Register(std::make_unique<mnncli::ListCommandHandler>());
                dispatcher.Register(std::make_unique<mnncli::SearchCommandHandler>());
                dispatcher.Register(std::make_unique<mnncli::DownloadCommandHandler>());
                dispatcher.Register(std::make_unique<mnncli::ModelInfoCommandHandler>());
                dispatcher.Register(std::make_unique<mnncli::ServeCommandHandler>());
                dispatcher.Register(std::make_unique<mnncli::BenchmarkCommandHandler>());
                dispatcher.Register(std::make_unique<mnncli::ConfigCommandHandler>());
                dispatcher.Register(std::make_unique<mnncli::InfoCommandHandler>());
                
                // Get spec for the command
                const auto& spec = mnncli::GetSpec(cmd_name);
                parser.SetCommandSpec(spec);
                
                // Parse the command
                auto cmd = parser.Parse();
                
                // Set verbose logging
                mnn::downloader::LogUtils::SetVerbose(cmd.verbose);
                
                // Dispatch to handler
                int result = dispatcher.Dispatch(cmd);
                if (result == -2) {
                    mnncli::UserInterface::ShowError("Internal error: handler not found");
                    return 1;
                }
                return result;
            } catch (const std::exception& e) {
                std::string err_msg = e.what();
                if (err_msg == "help_requested") {
                    const auto& spec = mnncli::GetSpec(cmd_name);
                    std::cout << mnncli::MakeUsage(spec) << "\n";
                    return 0;
                }
                mnncli::UserInterface::ShowError("Error: " + err_msg);
                return 1;
            }
        }
        
        // Legacy commands
        // Parse global options first
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-v" || arg == "--verbose") {
                verbose_ = true;
                mnn::downloader::LogUtils::SetVerbose(true);
                for (int j = i; j < argc - 1; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
                break;
            }
        }
        
        std::string cmd = argv[1];
        
        try {
            if (cmd == "--help" || cmd == "-h") {
                PrintUsage();
                return 0;
            } else if (cmd == "--version" || cmd == "-v") {
                PrintVersion();
                return 0;
            } else {
                mnncli::UserInterface::ShowError("Unknown command: " + cmd, "Use 'mnncli --help' for usage information");
                return 1;
            }
        } catch (const std::exception& e) {
            mnncli::UserInterface::ShowError("Unexpected error: " + std::string(e.what()));
            return 1;
        }
        
        return 0;
    }
    

    
private:
    
    void PrintUsage() {
        std::cout << "MNN CLI - AI Model Command Line Interface\n\n";
        std::cout << "Usage: mnncli <command> [options]\n\n";
        std::cout << "Commands:\n";
        std::cout << "  list       List local models\n";
        std::cout << "  search     Search remote models\n";
        std::cout << "  download   Download model\n";
        std::cout << "  delete     Delete model\n";
        std::cout << "  model_info Show model information, download location, and config content\n";
        std::cout << "  run        Run model inference\n";
        std::cout << "  serve      Start API server\n";
        std::cout << "  benchmark  Run performance benchmarks\n";
        std::cout << "  config     Manage configuration (show, set, reset, help)\n";
        std::cout << "  info       Show system information\n";
        std::cout << "\nGlobal Options:\n";
        std::cout << "  -v, --verbose  Enable verbose output for detailed debugging\n";
        std::cout << "  --help    Show this help message\n";
        std::cout << "  --version Show version information\n";
        std::cout << "\nExamples:\n";
        std::cout << "  mnncli list                          # List local models\n";
        std::cout << "  mnncli search qwen                   # Search for Qwen models\n";
        std::cout << "  mnncli download qwen-7b             # Download Qwen-7B model\n";
        std::cout << "  mnncli download qwen-7b -v          # Download with verbose output\n";
        std::cout << "  mnncli model_info qwen-7b           # Show model information and config\n";
        std::cout << "  mnncli model_info qwen-7b -v        # Show detailed model information\n";
        std::cout << "  mnncli config set download_provider modelscope  # Set default provider\n";
        std::cout << "  mnncli config show                   # Show current configuration\n";
        std::cout << "  mnncli config help                   # Show configuration help\n";
        std::cout << "  mnncli run                           # Run default model in interactive mode\n";
        std::cout << "  mnncli run qwen-7b                  # Run Qwen-7B model\n";
        std::cout << "  mnncli run -p \"Hello world\"         # Run with prompt using default model\n";
        std::cout << "  mnncli serve qwen-7b --port 8000    # Start API server\n";
        std::cout << "  mnncli benchmark qwen-7b            # Run benchmark\n";
    }
    
    
    void PrintVersion() {
        std::cout << "MNN CLI version 1.0.0\n";
        std::cout << "Built with MNN framework\n";
    }
    
    static bool IsR1(const std::string& path) {
        std::string lowerModelName = path;
        std::transform(lowerModelName.begin(), lowerModelName.end(), lowerModelName.begin(), ::tolower);
        return lowerModelName.find("deepseek-r1") != std::string::npos;
    }
    
    bool verbose_;
};

int main(int argc, const char* argv[]) {
    mnncli::UserInterface::ShowWelcome();
    
    CommandLineInterface cli;
    return cli.Run(argc, argv);
}
