# Contributing Guidelines

## Contribute

We welcome contributions from the community! By contributing, you can help improve FFRT and make it more robust.

Here are some ways you can contribute:

1. Report Issues:

    - If you encounter any bugs or have suggestions for improvements, please open an issue on this repository. Provide detailed information to help us understand and resolve the issue.

2. Submit Pull Requests:

    - If you have made improvements to the code, feel free to submit a pull request. Ensure that your code adheres to the projects's coding standards and is well-documented.
    - Please ensure that your code follows the project's coding standards. We use `.clang-format` to maintain consistent code style. Run the following command to format your code before committing:

        ```shell
        clang-format -i file
        ```

    - To ensure documentation quality, we use:

        - `cspell.json` for spell checking. Run the following command to check for spelling errors:

            ```shell
            cspell "src/**/*.cpp" "docs/**/*.md"
            ```

        - `.markdownlint.json` for Markdown file format standards. Run the following commands to lint your Markdown files:

            ```shell
            markdownlint "**/*.md"
            ```

        - `.autocorrectrc` and `.autocorrectignore` for ensuring proper formatting of mixed Chinese and English content. Run the following commands to autocorrect documentation:

            ```shell
            autocorrect "**/*.md"
            ```

    - We also recommend using the following VSCode plugins to help adhere to coding and documentation standards:

        - [Clang-Format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format) or [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) for maintaining consistent code style.
        - [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) for spell checking
        - [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint) for Markdown file format linting.
        - [AutoCorrect](https://marketplace.visualstudio.com/items?itemName=huacnlee.autocorrect) for auto-correcting mixed Chinese and English content.

3. Write Tests:

    - To maintain the quality of the project, we encourage you to write tests for new features and bug fixes. This helps ensure that the codebase remains stable and reliable.

4. Improve Documentation:

    - Good documentation is key to a successful project. If you find areas in the documentation that can be improved, please contribute by updating the `README.md` or other documentation files.

5. Review Pull Requests:

    - Reviewing and providing feedback on other contributor's pull requests is a valuable way to contribute. Your insights can help maintain the quality and consistency of the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](https://developer.huawei.com/consumer/en/devservice/guidelines/). Please read it to understand the standards and expectations for contributing.

## Get in Touch

If you have any questions or need further guidance, feel free to reach us via [email](mailto:hiffrt@huawei.com) or join our [Q&A community](https://developer.huawei.com/consumer/cn/forum/).

Thank you for your contributions and support!
