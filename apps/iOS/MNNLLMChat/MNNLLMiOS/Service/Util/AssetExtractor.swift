//
//  AssetExtractor.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/10.
//

import UIKit
import ImageIO
import Foundation

class AssetExtractor {
    
    static func copyFileToTmpDirectory(from sourceUrl: URL, fileName: String) -> URL? {
        let fileManager = FileManager.default
        let tempDirectory = fileManager.temporaryDirectory
        let destinationUrl = tempDirectory.appendingPathComponent(fileName)
        
        do {
            // Check if source file exists
            if !fileManager.fileExists(atPath: sourceUrl.path) {
                print("Source file does not exist at: \(sourceUrl.path)")
                return nil
            }

            // Check if the destination file already exists and remove it if needed
            if fileManager.fileExists(atPath: destinationUrl.path) {
                print("File already exists at: \(destinationUrl.path)")
                try fileManager.removeItem(at: destinationUrl)
            } else {
                // Attempt to copy the file
                try fileManager.copyItem(at: sourceUrl, to: destinationUrl)
                print("File successfully copied to: \(destinationUrl.path)")
            }
            return destinationUrl
        } catch {
            print("Error copying file: \(error.localizedDescription)")
            return nil
        }
    }
    
    /// Creates a local URL to extract an image from the Asset directory and save it to the cache directory.
    /// - Parameter name: The name of the image resource.
    /// - Returns: A local URL (if the image is successfully saved).
    static func createLocalUrl(forImageNamed name: String, withExtension fileExtension: String = "png") -> URL? {
        let fileManager = FileManager.default
        let cacheDirectory = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        
        let url = cacheDirectory.appendingPathComponent("\(name).\(fileExtension)")
        
        if fileManager.fileExists(atPath: url.path) {
            return url
        }
        
        guard let image = UIImage(named: name) else {
            print("Error: Image named \(name) not found in assets.")
            return nil
        }
        
        guard let data: Data = {
            if fileExtension.lowercased() == "png" {
                return image.pngData()
            } else if fileExtension.lowercased() == "jpg" || fileExtension.lowercased() == "jpeg" {
                return image.jpegData(compressionQuality: 1.0) // 高质量 JPEG
            } else {
                print("Error: Unsupported file extension \(fileExtension).")
                return nil
            }
        }() else {
            print("Error: Unable to convert image \(name) to data.")
            return nil
        }
        
        do {
            try data.write(to: url)
            print("File saved to: \(url.path)")
            return url
        } catch {
            print("Error: Unable to write file to \(url.path). Error: \(error)")
            return nil
        }
    }

    static func convertHEICToJPG(heicUrl: URL) -> URL? {
        guard let imageSource = CGImageSourceCreateWithURL(heicUrl as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            print("Failed to load HEIC image.")
            return nil
        }
        
        let properties = CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [String: Any]
        let orientation = properties?[kCGImagePropertyOrientation as String] as? UInt
        
        var uiImage = UIImage(cgImage: cgImage)
        
        if let orientation = orientation {
            uiImage = uiImage.imageRotatedByDegrees(degrees: orientation)
        }
        
        let jpegData = uiImage.jpegData(compressionQuality: 1.0)
        
        let jpgUrl = heicUrl.deletingPathExtension().appendingPathExtension("jpg")
        
        do {
            try jpegData?.write(to: jpgUrl)
            print("HEIC successfully converted to JPG and saved to: \(jpgUrl.path)")
            return jpgUrl
        } catch {
            print("Failed to save JPEG image: \(error.localizedDescription)")
            return nil
        }
    }
}


extension UIImage {
    func imageRotatedByDegrees(degrees: UInt) -> UIImage {
        var angle: CGFloat = 0.0
        
        switch degrees {
            case 1: angle = .pi / 2 // 90 degrees
            case 3: angle = .pi // 180 degrees
            case 6: angle = .pi / 2 // 90 degrees
            case 8: angle = -(.pi / 2) // -90 degrees
            default: break
        }
        
        UIGraphicsBeginImageContext(self.size)
        guard let context = UIGraphicsGetCurrentContext(), let cgImage = self.cgImage else {
            UIGraphicsEndImageContext()
            return self
        }
        context.translateBy(x: self.size.width / 2, y: self.size.height / 2)
        context.rotate(by: angle)
        context.scaleBy(x: 1.0, y: -1.0)
        context.draw(cgImage, in: CGRect(x: -self.size.width / 2, y: -self.size.height / 2, width: self.size.width, height: self.size.height))
        
        let rotatedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return rotatedImage ?? self
    }
}
