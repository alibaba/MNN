//
//  Created by Alex.M on 27.05.2022.
//

import SwiftUI

struct MediaCell: View {

    @StateObject var viewModel: MediaViewModel
    
    @Environment(\.mediaPickerTheme) private var theme

    var body: some View {
        ZStack {
            GeometryReader { geometry in
                ThumbnailView(preview: viewModel.preview)
                    .cornerRadius(theme.cellStyle.cornerRadius)
                    .onAppear {
                        viewModel.onStart(size: geometry.size)
                    }
            }
            .aspectRatio(1, contentMode: .fill)
            .clipped()

            if let duration = viewModel.assetMediaModel.asset.formattedDuration {
                VStack {
                    Spacer()
                    Rectangle()
                        .fill(LinearGradient(colors: [.black, .clear], startPoint: .bottom, endPoint: .top))
                }
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        Text(duration)
                            .font(.subheadline)
                            .foregroundColor(.white)
                            .padding(.trailing, 4)
                            .padding(.bottom, 4)
                    }
                }
            }
        }
        .onDisappear {
            viewModel.onStop()
        }
    }
}
