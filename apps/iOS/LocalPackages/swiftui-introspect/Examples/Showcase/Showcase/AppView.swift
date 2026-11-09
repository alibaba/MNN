import SwiftUI
import SwiftUIIntrospect

struct AppView: View {
    var body: some View {
        ContentView()
            #if os(iOS) || os(tvOS) || os(visionOS)
            .introspect(
                .window,
                on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
            ) { window in
                window.backgroundColor = .brown
            }
            #elseif os(macOS)
            .introspect(.window, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { window in
                window.backgroundColor = .lightGray
            }
            #endif
    }
}

struct ContentView: View {
    @State var selection = 0

    var body: some View {
        TabView(selection: $selection) {
            ListShowcase()
                .tabItem { Text("List") }
                .tag(0)
            ScrollViewShowcase()
                .tabItem { Text("ScrollView") }
                .tag(1)
            #if !os(macOS)
            NavigationShowcase()
                .tabItem { Text("Navigation") }
                .tag(2)
            PresentationShowcase()
                .tabItem { Text("Presentation") }
                .tag(3)
            #endif
            GenericViewShowcase()
                .tabItem { Text("Generic View") }
                .tag(4)
            SimpleElementsShowcase()
                .tabItem { Text("Simple elements") }
                .tag(5)
        }
        #if os(iOS) || os(tvOS)
        .introspect(.tabView, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) { tabBarController in
            tabBarController.tabBar.layer.backgroundColor = UIColor.green.cgColor
        }
        #elseif os(macOS)
        .introspect(.tabView, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { splitView in
            splitView.subviews.first?.layer?.backgroundColor = NSColor.green.cgColor
        }
        #endif
        .preferredColorScheme(.light)
    }
}

struct ListShowcase: View {
    var body: some View {
        VStack(spacing: 40) {
            VStack {
                Text("Default")
                    .lineLimit(1)
                    .minimumScaleFactor(0.5)
                    .padding(.horizontal, 12)
                List {
                    Text("Item 1")
                    Text("Item 2")
                }
            }

            VStack {
                Text(".introspect(.list, ...)")
                    .lineLimit(1)
                    .minimumScaleFactor(0.5)
                    .padding(.horizontal, 12)
                    .font(.system(.subheadline, design: .monospaced))
                List {
                    Text("Item 1")
                    Text("Item 2")
                }
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(.list, on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) { tableView in
                    tableView.backgroundView = UIView()
                    tableView.backgroundColor = .cyan
                }
                .introspect(.list, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2)) { collectionView in
                    collectionView.backgroundView = UIView()
                    collectionView.subviews.dropFirst(1).first?.backgroundColor = .cyan
                }
                #elseif os(macOS)
                .introspect(.list, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { tableView in
                    tableView.backgroundColor = .cyan
                }
                #endif
            }

            VStack {
                Text(".introspect(.list, ..., scope: .ancestor)")
                    .lineLimit(1)
                    .minimumScaleFactor(0.5)
                    .padding(.horizontal, 12)
                    .font(.system(.subheadline, design: .monospaced))
                List {
                    Text("Item 1")
                    Text("Item 2")
                        #if os(iOS) || os(tvOS) || os(visionOS)
                        .introspect(.list, on: .iOS(.v13, .v14, .v15), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), scope: .ancestor) { tableView in
                            tableView.backgroundView = UIView()
                            tableView.backgroundColor = .cyan
                        }
                        .introspect(.list, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), scope: .ancestor) { collectionView in
                            collectionView.backgroundView = UIView()
                            collectionView.subviews.dropFirst(1).first?.backgroundColor = .cyan
                        }
                        #elseif os(macOS)
                        .introspect(.list, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), scope: .ancestor) { tableView in
                            tableView.backgroundColor = .cyan
                        }
                        #endif
                }
            }
        }

    }
}

struct ScrollViewShowcase: View {
    var body: some View {
        VStack(spacing: 40) {
            ScrollView {
                Text("Default")
                    .frame(maxWidth: .infinity)
                    .lineLimit(1)
                    .minimumScaleFactor(0.5)
                    .padding(.horizontal, 12)
            }

            ScrollView {
                Text(".introspect(.scrollView, ...)")
                    .frame(maxWidth: .infinity)
                    .lineLimit(1)
                    .minimumScaleFactor(0.5)
                    .padding(.horizontal, 12)
                    .font(.system(.subheadline, design: .monospaced))
            }
            #if os(iOS) || os(tvOS) || os(visionOS)
            .introspect(
                .scrollView,
                on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
            ) { scrollView in
                scrollView.layer.backgroundColor = UIColor.cyan.cgColor
            }
            #elseif os(macOS)
            .introspect(.scrollView, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { scrollView in
                scrollView.drawsBackground = true
                scrollView.backgroundColor = .cyan
            }
            #endif

            ScrollView {
                Text(".introspect(.scrollView, ..., scope: .ancestor)")
                    .frame(maxWidth: .infinity)
                    .lineLimit(1)
                    .minimumScaleFactor(0.5)
                    .padding(.horizontal, 12)
                    .font(.system(.subheadline, design: .monospaced))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(
                        .scrollView,
                        on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2),
                        scope: .ancestor
                    ) { scrollView in
                        scrollView.layer.backgroundColor = UIColor.cyan.cgColor
                    }
                    #elseif os(macOS)
                    .introspect(.scrollView, on: .macOS(.v10_15, .v11, .v12, .v13, .v14), scope: .ancestor) { scrollView in
                        scrollView.drawsBackground = true
                        scrollView.backgroundColor = .cyan
                    }
                    #endif
            }
        }
    }
}

struct NavigationShowcase: View {
    var body: some View {
        NavigationView {
            Text("Content")
                .modifier {
                    if #available(iOS 15, tvOS 15, macOS 12, *) {
                        $0.searchable(text: .constant(""))
                    } else {
                        $0
                    }
                }
                #if os(iOS) || os(visionOS)
                .navigationBarTitle(Text("Customized"), displayMode: .inline)
                #elseif os(macOS)
                .navigationTitle(Text("Navigation"))
                #endif
        }
        #if os(iOS) || os(tvOS) || os(visionOS)
        .introspect(
            .navigationView(style: .stack),
            on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
        ) { navigationController in
            navigationController.navigationBar.backgroundColor = .cyan
        }
        .introspect(
            .navigationView(style: .columns),
            on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
        ) { splitViewController in
            #if os(visionOS)
            splitViewController.preferredDisplayMode = .oneBesideSecondary
            #else
            splitViewController.preferredDisplayMode = .oneOverSecondary
            #endif
        }
        .introspect(.navigationView(style: .columns), on: .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)) { navigationController in
            navigationController.navigationBar.backgroundColor = .cyan
        }
        .introspect(
            .searchField,
            on: .iOS(.v15, .v16, .v17, .v18), .tvOS(.v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
        ) { searchBar in
            searchBar.backgroundColor = .red
            #if os(iOS)
            searchBar.searchTextField.backgroundColor = .purple
            #endif
        }
        #endif
    }
}

#if !os(macOS)
struct PresentationShowcase: View {
    @State var isSheetPresented = false
    @State var isFullScreenPresented = false
    @State var isPopoverPresented = false

    var body: some View {
        VStack(spacing: 20) {
            Button("Sheet", action: { isSheetPresented = true })
                .sheet(isPresented: $isSheetPresented) {
                    Button("Dismiss", action: { isSheetPresented = false })
                        #if os(iOS) || os(tvOS)
                        .introspect(
                            .sheet,
                            on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18)
                        ) { presentationController in
                            presentationController.containerView?.backgroundColor = .red.withAlphaComponent(0.75)
                        }
                        #elseif os(visionOS)
                        .introspect(.sheet, on: .visionOS(.v1, .v2)) { sheetPresentationController in
                            sheetPresentationController.containerView?.backgroundColor = .red.withAlphaComponent(0.75)
                        }
                        #endif
                }

            if #available(iOS 14, tvOS 14, *) {
                Button("Full Screen Cover", action: { isFullScreenPresented = true })
                    .fullScreenCover(isPresented: $isFullScreenPresented) {
                        Button("Dismiss", action: { isFullScreenPresented = false })
                            #if os(iOS) || os(tvOS) || os(visionOS)
                            .introspect(
                                .fullScreenCover,
                                on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
                            ) { presentationController in
                                presentationController.containerView?.backgroundColor = .red.withAlphaComponent(0.75)
                            }
                            #endif
                    }
            }

            #if os(iOS) || os(visionOS)
            Button("Popover", action: { isPopoverPresented = true })
                .popover(isPresented: $isPopoverPresented) {
                    Button("Dismiss", action: { isPopoverPresented = false })
                        .padding()
                        .introspect(
                            .popover,
                            on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
                        ) { presentationController in
                            presentationController.containerView?.backgroundColor = .red.withAlphaComponent(0.75)
                        }
                }
            #endif
        }
    }
}
#endif

struct GenericViewShowcase: View {
    var body: some View {
        VStack(spacing: 10) {
            Text(".introspect(.view, ...)")
                .lineLimit(1)
                .minimumScaleFactor(0.5)
                .font(.system(.subheadline, design: .monospaced))
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(
                    .view,
                    on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
                ) { view in
                    view.backgroundColor = .cyan
                }
                #elseif os(macOS)
                .introspect(.view, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { view in
                    view.layer?.backgroundColor = NSColor.cyan.cgColor
                }
                #endif

            Button("A button", action: {})
                .padding(5)
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(
                    .view,
                    on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
                ) { view in
                    view.backgroundColor = .lightGray
                }
                #elseif os(macOS)
                .introspect(.view, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { view in
                    view.layer?.backgroundColor = NSColor.lightGray.cgColor
                }
                #endif

            Image(systemName: "scribble")
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(
                    .view,
                    on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
                ) { view in
                    view.backgroundColor = .blue
                }
                #elseif os(macOS)
                .introspect(.view, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { view in
                    view.layer?.backgroundColor = NSColor.blue.cgColor
                }
                #endif
        }
        .padding()
        #if os(iOS) || os(tvOS) || os(visionOS)
        .introspect(
            .view,
            on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
        ) { view in
            view.backgroundColor = .red
        }
        #elseif os(macOS)
        .introspect(.view, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { view in
            view.layer?.backgroundColor = NSColor.red.cgColor
        }
        #endif
    }
}

struct SimpleElementsShowcase: View {

    @State private var textFieldValue = ""
    @State private var toggleValue = false
    @State private var sliderValue = 0.0
    @State private var datePickerValue = Date()
    @State private var segmentedControlValue = 0

    var body: some View {
        VStack {
            HStack {
                TextField("Text Field Red", text: $textFieldValue)
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(
                        .textField,
                        on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
                    ) { textField in
                        textField.backgroundColor = .red
                    }
                    #elseif os(macOS)
                    .introspect(.textField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { textField in
                        textField.backgroundColor = .red
                    }
                    #endif

                TextField("Text Field Green", text: $textFieldValue)
                    .cornerRadius(8)
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(
                        .textField,
                        on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
                    ) { textField in
                        textField.backgroundColor = .green
                    }
                    #elseif os(macOS)
                    .introspect(.textField, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { textField in
                        textField.backgroundColor = .green
                    }
                    #endif
            }

            #if !os(tvOS)
            #if !os(visionOS)
            HStack {
                Toggle("Toggle Red", isOn: $toggleValue)
                    #if os(iOS)
                    .introspect(
                        .toggle,
                        on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)
                    ) { toggle in
                        toggle.backgroundColor = .red
                    }
                    #elseif os(macOS)
                    .introspect(.toggle, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { toggle in
                        toggle.layer?.backgroundColor = NSColor.red.cgColor
                    }
                    #endif

                Toggle("Toggle Green", isOn: $toggleValue)
                    #if os(iOS)
                    .introspect(
                        .toggle,
                        on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)
                    ) { toggle in
                        toggle.backgroundColor = .green
                    }
                    #elseif os(macOS)
                    .introspect(.toggle, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { toggle in
                        toggle.layer?.backgroundColor = NSColor.green.cgColor
                    }
                    #endif
            }

            HStack {
                Slider(value: $sliderValue, in: 0...100)
                    #if os(iOS)
                    .introspect(.slider, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) { slider in
                        slider.backgroundColor = .red
                    }
                    #elseif os(macOS)
                    .introspect(.slider, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { slider in
                        slider.layer?.backgroundColor = NSColor.red.cgColor
                    }
                    #endif

                Slider(value: $sliderValue, in: 0...100)
                    #if os(iOS)
                    .introspect(.slider, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) { slider in
                        slider.backgroundColor = .green
                    }
                    #elseif os(macOS)
                    .introspect(.slider, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { slider in
                        slider.layer?.backgroundColor = NSColor.green.cgColor
                    }
                    #endif
            }

            HStack {
                Stepper(onIncrement: {}, onDecrement: {}) {
                    Text("Stepper Red")
                }
                #if os(iOS)
                .introspect(.stepper, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) { stepper in
                    stepper.backgroundColor = .red
                }
                #elseif os(macOS)
                .introspect(.stepper, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { stepper in
                    stepper.layer?.backgroundColor = NSColor.red.cgColor
                }
                #endif

                Stepper(onIncrement: {}, onDecrement: {}) {
                    Text("Stepper Green")
                }
                #if os(iOS)
                .introspect(.stepper, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) { stepper in
                    stepper.backgroundColor = .green
                }
                #elseif os(macOS)
                .introspect(.stepper, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { stepper in
                    stepper.layer?.backgroundColor = NSColor.green.cgColor
                }
                #endif
            }
            #endif

            HStack {
                DatePicker(selection: $datePickerValue) {
                    Text("DatePicker Red")
                }
                #if os(iOS) || os(visionOS)
                .introspect(.datePicker, on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)) { datePicker in
                    datePicker.backgroundColor = .red
                }
                #elseif os(macOS)
                .introspect(.datePicker, on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { datePicker in
                    datePicker.layer?.backgroundColor = NSColor.red.cgColor
                }
                #endif
            }
            #endif

            HStack {
                Picker(selection: $segmentedControlValue, label: Text("Segmented control")) {
                    Text("Option 1").tag(0)
                    Text("Option 2").tag(1)
                    Text("Option 3").tag(2)
                }
                .pickerStyle(SegmentedPickerStyle())
                #if os(iOS) || os(tvOS) || os(visionOS)
                .introspect(
                    .picker(style: .segmented),
                    on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2)
                ) { datePicker in
                    datePicker.backgroundColor = .red
                }
                #elseif os(macOS)
                .introspect(.picker(style: .segmented), on: .macOS(.v10_15, .v11, .v12, .v13, .v14)) { datePicker in
                    datePicker.layer?.backgroundColor = NSColor.red.cgColor
                }
                #endif
            }
        }

    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
