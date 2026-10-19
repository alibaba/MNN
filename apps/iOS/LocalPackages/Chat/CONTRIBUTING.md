# Contributing to ExyteChat

## Code Style

Please use selected code formatting style:
- 4 spaces tab
- no spaces on empty lines
- comment: "// start with small letter"
- declaration: "var users: [User]"

### swift-format

You can use [swift-format](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjol7H-_6iLAxW5ZmwGHViJIX8QFnoECBoQAQ&url=https%3A%2F%2Fgithub.com%2Fswiftlang%2Fswift-format&usg=AOvVaw0kMi_vMj0IW_Vm5BZ8ffcT&opi=89978449) to do this

Code style is specified in .swift-format
Run swift-format before checkin to ensure matching code style

```bash
#example swift format command
swift-format format -i --configuration .swift-format Sources/ExyteChat/ChatView/ChatView.swift
```




