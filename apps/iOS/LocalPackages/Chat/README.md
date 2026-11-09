<a href="https://exyte.com/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/exyte/media/master/common/header-dark.png"><img src="https://raw.githubusercontent.com/exyte/media/master/common/header-light.png"></picture></a>

<a href="https://exyte.com/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/exyte/media/master/common/our-site-dark.png" width="80" height="16"><img src="https://raw.githubusercontent.com/exyte/media/master/common/our-site-light.png" width="80" height="16"></picture></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://twitter.com/exyteHQ"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/exyte/media/master/common/twitter-dark.png" width="74" height="16"><img src="https://raw.githubusercontent.com/exyte/media/master/common/twitter-light.png" width="74" height="16">
</picture></a> <a href="https://exyte.com/contacts"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/exyte/media/master/common/get-in-touch-dark.png" width="128" height="24" align="right"><img src="https://raw.githubusercontent.com/exyte/media/master/common/get-in-touch-light.png" width="128" height="24" align="right"></picture></a>

<table>
    <thead>
        <tr>
            <th>Chat</th>
            <th>Media</th>
            <th>Audio Messages</th>
            <th>Extra</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
                <img src="https://github.com/exyte/Chat/assets/1358172/baf0167f-b3e0-4df2-bd3b-b6b1c4ee385d" />
            </td>
            <td>
                <img src="https://github.com/exyte/Chat/assets/1358172/d62876ef-4475-4f07-933a-9d9366b02e28" />
            </td>
            <td>
                <img src="https://github.com/exyte/Chat/assets/1358172/ebd2040d-1cf0-4066-9391-592af1426571" />
            </td>
            <td>
                <img src="https://github.com/exyte/Chat/assets/1358172/053bcd73-0db7-44da-abd6-0a57f0f88a4b" />
            </td>
        </tr>
    </tbody>
</table>

<p><h1>Chat</h1></p>
<p><h4>A SwiftUI Chat UI framework with fully customizable message cells and a built-in media picker</h4></p>

![](https://img.shields.io/github/v/tag/exyte/Chat?label=Version)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fexyte%2FChat%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/exyte/Chat)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fexyte%2FChat%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/exyte/Chat)
[![SPM Compatible](https://img.shields.io/badge/SwiftPM-Compatible-brightgreen.svg)](https://swiftpackageindex.com/exyte/Chat)
[![Cocoapods Compatible](https://img.shields.io/badge/cocoapods-Compatible-brightgreen.svg)](https://cocoapods.org/pods/ExyteChat)
[![Carthage Compatible](https://img.shields.io/badge/Carthage-compatible-brightgreen.svg?style=flat)](https://github.com/Carthage/Carthage)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/licenses/MIT)

# Features
- Displays your messages with pagination and allows you to create and "send" new messages (sending means calling a closure since user will be the one providing actual API calls)
- Allows you to pass a custom view builder for messages and input views
- Has a built-in photo and video library/camera picker for multiple media asset selection
- Can display a fullscreen menu on long press a message cell (automatically shows scroll for big messages)
- Supports "reply to message" via message menu or through a closure. Remove and edit are **coming soon**
- This library allows to send the following content in messages in any combination:
    - Text with/without markdown
    - Photo/video
    - Audio recording
    **Coming soon:**
    - User's location
    - Documents
    - Link with preview

# Usage

Create a chat view like this:
```swift
@State var messages: [Message] = []

var body: some View {
    ChatView(messages: messages) { draft in
        yourViewModel.send(draft: draft)
    }
}
```
where:  
   `messages` - list of messages to display  
   `didSendMessage` - a closure which is called when the user presses the send button  

`Message` is a type that `Chat` uses for the internal implementation. In the code above it expects the user to provide a list of `Message` structs, and it returns a `DraftMessage` in the `didSendMessage` closure. You can map it both ways to your own `Message` model that your API expects or use as is.

## Available chat types
Chat type - determines the order of messages and direction of new message animation. Available options:
- `conversation` - the latest message is at the bottom, new messages appear from the bottom  
- `comments` - the latest message is at the top, new messages appear from the top  

Reply mode - determines how replying to message looks. Available options:
- `quote` - when replying to message A, new message will appear as the newest message, quoting message A in its body  
- `answer` - when replying to message A, new message with appear direclty below message A as a separate cell without duplicating message A in its body  

To specify any of these pass them through `init`:
```swift
ChatView(messages: viewModel.messages, chatType: .comments, replyMode: .answer) { draft in
    yourViewModel.send(draft: draft)
}
```

## Custom UI
You may customize message cells like this: 
```swift
ChatView(messages: viewModel.messages) { draft in
    viewModel.send(draft: draft)
} messageBuilder: { message, positionInUserGroup, positionInCommentsGroup, showContextMenuClosure, messageActionClosure, showAttachmentClosure in
    VStack {
        Text(message.text)
        if !message.attachments.isEmpty {
            ForEach(message.attachments, id: \.id) { at in
                AsyncImage(url: at.thumbnail)
            }
        }
    }
}
```
`messageBuilder`'s parameters:  
- `message` - the message containing user info, attachments, etc.   
- `positionInUserGroup` - the position of the message in its continuous collection of messages from the same user    
- `positionInCommentsGroup` - position of message in its continuous group of comments (only works for .answer ReplyMode, nil for .quote mode)  
- `showContextMenuClosure` - closure to show message context menu   
- `messageActionClosure ` - closure to pass user interaction, .reply for example   
- `showAttachmentClosure` - you can pass an attachment to this closure to use ChatView's fullscreen media viewer    

You may customize the input view (a text field with buttons at the bottom) like this: 
```swift
ChatView(messages: viewModel.messages) { draft in
    viewModel.send(draft: draft)
} inputViewBuilder: { textBinding, attachments, inputViewState, inputViewStyle, inputViewActionClosure, dismissKeyboardClosure in
    Group {
        switch inputViewStyle {
        case .message: // input view on chat screen
            VStack {
                HStack {
                    Button("Send") { inputViewActionClosure(.send) }
                    Button("Attach") { inputViewActionClosure(.photo) }
                }
                TextField("Write your message", text: textBinding)
            }
        case .signature: // input view on photo selection screen
            VStack {
                HStack {
                    Button("Send") { inputViewActionClosure(.send) }
                }
                TextField("Compose a signature for photo", text: textBinding)
                    .background(Color.green)
            }
        }
    }
}
```
`inputViewBuilder`'s parameters:  
- `textBinding` to bind your own TextField   
- `attachments` is a struct containing photos, videos, recordings and a message you are replying to     
- `inputViewState` - the state of the input view that is controlled by the library automatically if possible or through your calls of `inputViewActionClosure`
- `inputViewStyle` - `.message` or `.signature` (the chat screen or the photo selection screen)   
- `inputViewActionClosure` for calling on taps on your custom buttons. For example, call `inputViewActionClosure(.send)` if you want to send your message with your own button, then the library will reset the text and attachments and call the `didSendMessage` sending closure   
- `dismissKeyboardClosure` - call this to dismiss keyboard    

## Custom message menu
Long tap on a message will display a menu for this message (can be turned off, see Modifiers). To define custom message menu actions declare an enum conforming to `MessageMenuAction`. Then the library will show your custom menu options on long tap on message instead of default ones, if you pass your enum's name to it (see code sample). Once the action is selected special callbcak will be called. Here is a simple example:
```swift
enum Action: MessageMenuAction {
    case reply, edit

    func title() -> String {
        switch self {
        case .reply:
            "Reply"
        case .edit:
            "Edit"
        }
    }
    
    func icon() -> Image {
        switch self {
        case .reply:
            Image(systemName: "arrowshape.turn.up.left")
        case .edit:
            Image(systemName: "square.and.pencil")
        }
    }
}

ChatView(messages: viewModel.messages) { draft in
    viewModel.send(draft: draft)
} messageMenuAction: { (action: Action, defaultActionClosure, message) in // <-- here: specify the name of your `MessageMenuAction` enum
    switch action {
    case .reply:
        defaultActionClosure(message, .reply)
    case .edit:
        defaultActionClosure(message, .edit { editedText in
            // update this message's text on your BE
            print(editedText)
        })
    }
}
```
`messageMenuAction`'s parameters:  
- `selectedMenuAction` - action selected by the user from the menu. NOTE: when declaring this variable, specify its type (your custom descendant of MessageMenuAction) explicitly    
- `defaultActionClosure` - a closure taking a case of default implementation of MessageMenuAction which provides simple actions handlers; you call this closure passing the selected message and choosing one of the default actions (.reply, .edit) if you need them; or you can write a custom implementation for all your actions, in that case just ignore this closure
- `message` - message for which the menu is displayed
    
When implementing your own `MessageMenuActionClosure`, write a switch statement passing through all the cases of your `MessageMenuAction`, inside each case write your own action handler, or call the default one. NOTE: not all default actions work out of the box - e.g. for `.edit` you'll still need to provide a closure to save the edited text on your BE. Please see CommentsExampleView in ChatExample project for MessageMenuActionClosure usage example.

## Small view builders:
These use `AnyView`, so please try to keep them easy enough
- `betweenListAndInputViewBuilder` - content to display in between the chat list view and the input view   
- `mainHeaderBuilder` - a header for the whole chat, which will scroll together with all the messages and headers  
- `headerBuilder` - date section header builder   

## Modifiers 
`isListAboveInputView` - messages table above the input field view or not   
`showDateHeaders` - show section headers with dates between days, default is `true`    
`isScrollEnabled` - forbid scrolling for messages' `UITabelView`   
`showMessageMenuOnLongPress` - turn menu on long tap on/off    
`showNetworkConnectionProblem` - display network error on/off    
`assetsPickerLimit` - set a limit for MediaPicker built into the library   
`setMediaPickerSelectionParameters` - a struct holding MediaPicker selection parameters (assetsPickerLimit and others like mediaType, selectionStyle, etc.).   
`orientationHandler` - handle screen rotation

`enableLoadMore(offset: Int, handler: @escaping ChatPaginationClosure)` - when user scrolls to `offset`-th message from the end, call the handler function, so the user can load more messages. NOTE: New messages won't appear in the chat unless it's scrolled up to the very top - it's an optimization. 

### Customize default UI
You can use `chatTheme` to customize colors and images of default UI. You can pass all/some colors and images:

```swift
.chatTheme(
    ChatTheme(
        colors: .init(
            mainBackground: .red,
            buttonBackground: .yellow,
            addButtonBackground: .purple
        ),
        images: .init(
            camera: Image(systemName: "camera")
        )
    )
)
```
Please use `mediaPickerTheme` in a similar fashion to customize the built-in photo picker.      

### makes sense only for built-in message view
`avatarSize` - the default avatar is a circle, you can specify its diameter here 
`tapAvatarClosure` - closure to call on avatar tap   
`messageUseMarkdown` - use markdown (e.g. ** to make something bold) or not
`showMessageTimeView` - show timestamp in a corner of the message   
`setMessageFont` - pass custom font to use for messages   

### makes sense only for built-in input view
`setAvailableInput` - hide some buttons in default InputView. Available options are:
    - `.full` - media + text + audio   
    - `.textAndMedia`   
    - `.textAndAudio`   
    - `.textOnly`    
  
<img src="https://raw.githubusercontent.com/exyte/media/master/Chat/pic2.png" width="300">

## Localization

You can localize the inputs using the standard SwiftUI localization process, add the input strings to each languages Localizable.strings file.  
The library uses the following text that can be localized:

- Type a message...
- Add signature...
- Cancel
- Recents
- Waiting for network
- Recording...
- Reply to

## Examples
There are 2 example projects:    
- One has a simple bot posting random text/media messages every 2 seconds. It has no back end and no local storage. Every new start is clean and fresh.     
- Another has an integration with Firestore data base. It has all the necessary back end support, including storing media and audio messages, unread messages counters, etc. You'll have to create your own Firestore app and DB. Also replace `GoogleService-Info` with your own. After that you can test on multiple sims/devices.    

Create your firestore app
https://console.firebase.google.com/
Create firesote database (for light weight text data)
https://firebase.google.com/docs/firestore/manage-data/add-data
Create cloud firestore database (for images and voice recordings)
https://firebase.google.com/docs/storage/web/start

## Example

To try out the Chat examples:
- Clone the repo `git clone git@github.com:exyte/Chat.git`
- Open terminal and run `cd <ChatRepo>/Example`
- Wait for SPM to finish downloading packages
- Run it!

## Installation

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/exyte/Chat.git")
]
```

### CocoaPods

```ruby
pod 'ExyteChat'
```

### Carthage

```ogdl
github "Exyte/Chat"
```

## Requirements

* iOS 16+
* Xcode 14+

## Our other open source SwiftUI libraries
[PopupView](https://github.com/exyte/PopupView) - Toasts and popups library    
[Grid](https://github.com/exyte/Grid) - The most powerful Grid container    
[ScalingHeaderScrollView](https://github.com/exyte/ScalingHeaderScrollView) - A scroll view with a sticky header which shrinks as you scroll    
[AnimatedTabBar](https://github.com/exyte/AnimatedTabBar) - A tabbar with a number of preset animations   
[MediaPicker](https://github.com/exyte/mediapicker) - Customizable media picker     
[OpenAI](https://github.com/exyte/OpenAI) Wrapper lib for [OpenAI REST API](https://platform.openai.com/docs/api-reference/introduction)    
[AnimatedGradient](https://github.com/exyte/AnimatedGradient) - Animated linear gradient     
[ConcentricOnboarding](https://github.com/exyte/ConcentricOnboarding) - Animated onboarding flow    
[FloatingButton](https://github.com/exyte/FloatingButton) - Floating button menu    
[ActivityIndicatorView](https://github.com/exyte/ActivityIndicatorView) - A number of animated loading indicators    
[ProgressIndicatorView](https://github.com/exyte/ProgressIndicatorView) - A number of animated progress indicators    
[FlagAndCountryCode](https://github.com/exyte/FlagAndCountryCode) - Phone codes and flags for every country    
[SVGView](https://github.com/exyte/SVGView) - SVG parser    
[LiquidSwipe](https://github.com/exyte/LiquidSwipe) - Liquid navigation animation    

