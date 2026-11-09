//
//  SquadDataset.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation

/// Our internal type, also used in unit tests
struct SquadExample {
    let qaId: String
    let context: String
    let question: String
    let answerText: String
    let startPos: Int
    let endPos: Int
}

/// Types from the actual squad datasets
struct SquadDataset: Codable {
    let data: [SquadDatum]
    let version: String
}

struct SquadDatum: Codable {
    let paragraphs: [SquadParagraph]
    let title: String
}

struct SquadParagraph: Codable {
    let context: String
    let qas: [SquadQA]
}

struct SquadQA: Codable {
    let answers: [SquadAnswer]
    let id: String
    let question: String
}

struct SquadAnswer: Codable {
    let answer_start: Int
    let text: String
}

struct Squad {
    /// Get all examples from the Squad dataset.
    static let examples: [SquadExample] = {
        let url = Bundle.module.url(forResource: "dev-v1.1", withExtension: "json")!
        let json = try! Data(contentsOf: url)
        let decoder = JSONDecoder()
        let squadDataset = try! decoder.decode(SquadDataset.self, from: json)

        var examples: [SquadExample] = []
        for datum in squadDataset.data {
            for paragraph in datum.paragraphs {
                for qa in paragraph.qas {
                    let example = SquadExample(qaId: qa.id, context: paragraph.context, question: qa.question, answerText: qa.answers[0].text, startPos: qa.answers[0].answer_start, endPos: -1) // TODO: remove -1
                    examples.append(example)
                }
            }
        }
        return examples
    }()
}
