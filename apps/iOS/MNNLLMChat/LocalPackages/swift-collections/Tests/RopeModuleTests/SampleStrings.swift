//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2023 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#if !COLLECTIONS_SINGLE_MODULE
import _CollectionsTestSupport
#endif

func randomStride(
  from start: Int,
  to end: Int,
  by maxStep: Int,
  seed: Int
) -> UnfoldSequence<Int, Int> {
  var rng = RepeatableRandomNumberGenerator(seed: seed)
  return sequence(state: start, next: {
    $0 += Int.random(in: 1 ... maxStep, using: &rng)
    guard $0 < end else { return nil }
    return $0
  })
}

let sampleString: String = {
  var str = #"""
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    一种强大但极易学习的编程语言。
    Swift 是一种强大直观的编程语言，适用于 iOS、iPadOS、macOS、Apple tvOS 和 watchOS。\#
    编写 Swift 代码的过程充满了乐趣和互动。Swift 语法简洁，但表现力强，更包含了开发者喜爱的现代功能。\#
    Swift 代码从设计上保证安全，并能开发出运行快如闪电的软件。
    
    パワフルなプログラミング言語でありながら、簡単に習得することができます。
    Swiftは、iOS、iPadOS、macOS、tvOS、watchOS向けのパワフルで直感的なプログラミング言語です。\#
    Swiftのコーディングはインタラクティブで楽しく、構文はシンプルでいて表現力に富んでいます。\#
    さらに、デベロッパが求める最新の機能も備えています。安全性を重視しつつ非常に軽快に動作す\#
    るソフトウェアを作り出すことができます。それがSwiftです。
    
    손쉽게 학습할 수 있는 강력한 프로그래밍 언어.
    Swift는 iOS, iPadOS, macOS, tvOS 및 watchOS를 위한 강력하고 직관적인 프로그래밍 언어입니다. \#
    Swift 코드 작성은 대화식으로 재미있고, 구문은 간결하면서도 표현력이 풍부하며, Swift에는 개발자들이 \#
    좋아하는 첨단 기능이 포함되어 있습니다. Swift 코드는 안전하게 설계되었으며 빛의 속도로 빠르게 실행되는 \#
    소프트웨어를 제작할 수 있습니다.
    
    🪙 A 🥞 short 🍰 piece 🫘 of 🌰 text 👨‍👨‍👧‍👧 with 👨‍👩‍👦 some 🚶🏽 emoji 🇺🇸🇨🇦 characters 🧈
    some🔩times 🛺 placed 🎣 in 🥌 the 🆘 mid🔀dle 🇦🇶or🏁 around 🏳️‍🌈 a 🍇 w🍑o🥒r🥨d
    
    ⌘⏎ ⌃⇧⌥⌘W
    ¯\_(ツ)_/¯
    ಠ_ಠ
    
    🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦
    """#
  if MemoryLayout<Int>.size == 8 {
    /// Add even more flags and extra long combining sequences. This considerably increases test
    /// workload. (Not necessarily due to grapheme cluster length, but because test performance is
    /// quadratic or worse in the size of the overall text.)
    str += #"""
    🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦\#
    🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦🇺🇸🇨🇦

    Unicode is such fun!
    U̷n̷i̷c̷o̴d̴e̷ ̶i̸s̷ ̸s̵u̵c̸h̷ ̸f̵u̷n̴!̵
    U̴̡̲͋̾n̵̻̳͌ì̶̠̕c̴̭̈͘ǫ̷̯͋̊d̸͖̩̈̈́ḛ̴́ ̴̟͎͐̈i̴̦̓s̴̜̱͘ ̶̲̮̚s̶̙̞͘u̵͕̯̎̽c̵̛͕̜̓h̶̘̍̽ ̸̜̞̿f̵̤̽ṷ̴͇̎͘ń̷͓̒!̷͍̾̚
    U̷̢̢̧̨̼̬̰̪͓̞̠͔̗̼̙͕͕̭̻̗̮̮̥̣͉̫͉̬̲̺͍̺͊̂ͅn̶̨̢̨̯͓̹̝̲̣̖̞̼̺̬̤̝̊̌́̑̋̋͜͝ͅḭ̸̦̺̺͉̳͎́͑c̵̛̘̥̮̙̥̟̘̝͙̤̮͉͔̭̺̺̅̀̽̒̽̏̊̆͒͌̂͌̌̓̈́̐̔̿̂͑͠͝͝ͅö̶̱̠̱̤̙͚͖̳̜̰̹̖̣̻͎͉̞̫̬̯͕̝͔̝̟̘͔̙̪̭̲́̆̂͑̌͂̉̀̓́̏̎̋͗͛͆̌̽͌̄̎̚͝͝͝͝ͅd̶̨̨̡̡͙̟͉̱̗̝͙͍̮͍̘̮͔͑e̶̢͕̦̜͔̘̘̝͈̪̖̺̥̺̹͉͎͈̫̯̯̻͑͑̿̽͂̀̽͋́̎̈́̈̿͆̿̒̈́̽̔̇͐͛̀̓͆̏̾̀̌̈́̆̽̕ͅ ̷̢̳̫̣̼̙̯̤̬̥̱͓̹͇̽̄̄̋̿̐̇̌̒̾̑̆̈́̏͐̒̈̋̎͐̿̽̆̉͋͊̀̍͘̕̕̕͝͠͠͝ͅͅì̸̢̧̨̨̮͇̤͍̭̜̗̪̪͖̭͇͔̜̗͈̫̩͔̗͔̜̖̲̱͍̗̱̩͍̘̜̙̩͔̏̋̓̊́́̋̐̌͊͘̕͠s̶̨̢̧̥̲̖̝̩͖̱͋́͑͐̇̐̔̀̉͒͒́̐̉̔͘͠͠ ̵̧̛͕̦̭̣̝̩͕̠͎̮͓͉̟̠̘͎͋͗͆̋̌̓̃̏̊̔̾̒̿s̸̟͚̪̘̰̮͉̖̝̅̓͛̏̆ư̵͍̙̠͍̜͖͔̮̠̦̤̣̯̘̲͍͂͌̌̅̍͌̈́̆̋̎͋̓̍͆̃̑͌͘̕͜ͅç̸̟̗͉̟̤̙̹͓̖͇̳̈́̍̏͐̓̓̈̆̉̈͆̍ͅh̵̛̛̹̪͇͓̤̺̟͙̣̰͓̺̩̤̘̫͔̺͙͌́̑̓͗̏͆́͊̈́̋̿͒̐̀́̌͜͜͝ ̴̗͓͚͖̣̥͛́̓͐͂͛̐͑̈́͗̂̈͠f̶̡̩̟̤̭̩̱̥͈̼̥̳͕̣͓̱̰͎̖̦͎̦̻̫͉̝̗̝͚̎͌͑̾̿̊̉͆̉̏̅̔̓̈́̀͐̚͘ͅư̷̦̮͖͙̺̱̼̜̺̤͎̜͐͐̊̊̈͋̔̓̍͊̇̊̈́̈͑̐̎̿̑̋͋̀̅̓͛̚͜n̷̡̨͉̠̖̙͎̳̠̦̼̻̲̳̿̀̓̍͋̎͆̓̇̾̅͊̐͘͘̕!̷̡̨̧̢̡̡̼̹̭̝̝̭̫̫̥̰̤̪̦̤̼̖̖̳̰̲͙͕̖̬̳̪͖̹̮͐͐͊̈́̐͑͛̾̈͊̊͋͑̉͒̈̿̈̃̑͋͐́͊̀͝͠͝͠
    Ų̷̧̧̧̧̡̢̨̨̢̢̢̧̡̨̨̧̢̡̡̢̨̨̮̜͈̳̮͔̺͚̹͉͍̫̪͖͙͙̳͖͖̦̮̜̫̗̣̙̪͇̩̻̬̖̝̻̰͙̖̙̭̤͎̠͇̹̦̤̟̦͎̹̝̗̫͔̳̣̦͍̹͈̺̮͈͈̬̭̘͕̟͉̮̟͖̦̥͇̠̙̳̲̝̦͖̻̪̺̬̫͈͈̘͍͚͖̝̥̙͖̪͔̫̣̘͙͓̱̠̲̯̦͓͖͚͎͉͖̘̺͕͇̱̺̗̙̮̮̹̯̤̮̺͓̘̫͕̞̮͕̠̗͍̦̣̮͙͉͈̭̜̭̘̼̼̖̮̘̝͈̌̑͋͂̈́̐̄̂̆̊̈̅͆͋̔͗̍͒̆̐̒̽͑̋́͛͗̓̃̽͋̒͑̈́̕̕̚̕̕͜͜͝͠͝͠͝ͅͅņ̴̡̢̡̢̢̨̛̛̛̛̛̛̛̻̬̲̰̗̭͕̯͇̩̦̮̫̭̰̪͉̹̭͇̣̦͕̹̗̭̬͓͕͍̯͇͕̩̱̲͍̟̙͓̣̖̱͍̟͚͔̞̪͕̣̺̻̭̖̤̜͈̰̻̘̹̝̝̮̗͔̯̻̻͍͕̬͇͓̲̗̟̭̰̬̳͈̼̤̙̱̻̜͍̪̣̈̉̉͑̇̇͐̆̆̀̋̄̂̿͒͐͒͛̒̍͆̿͐͊̂̿̐̇̋̄̈̂̓̅̇̈́̾̒̔̍̈́͐̋͊̑͐͆̈́̿́̽͛́̊̏̓̾̇̈́̀̃͐̃̈́́͒̏̀͑̑̅̈́̇͂̓̐̒́̾̈́͗̋̅̀́͋̍͑͒̌̔̈́̆̂̉͌̈́̑̾̑̽̓̓̏̄͋̽̒̓͌̊̂̀͑̂͑̌͋̓͐̈̃̾̏̃̏͑̋̒̈̀̔͐̓̋̉̐̐͋͆̂̈́̒͊̏̓̔͌̈̾͗͑͂̾͆͑̂̍̂͗͆̄̊̐̏̈́͂̌͐̃̐͌̊̀͗̐͑̔͋͒̊̋̒̐̄͐̏̓͌̃̌͋̔̎̈̆̍̃̿́̇̉̀́̊̅̌̏̆̆͛̔͋̈̽̂͂̇̓́͛͗̔̍̃̈̽͑͐̿̽̉̓̎̔͗̊͂̽͘̕̚̕͘̕̕͘̚͘̚̕̚̚̚͘̚͜͜͝͝͠͝͝͠͝͝͝͝͝͝͝͝͝͝͠͝ͅͅͅi̴̧̡̢̡̧̢̢̢̧̡̧̢̡̹̤̗̭̭̭̪̳̮͉̦̪͉͈̦̗̣̼̻̜̰̳̯͕̩̘̙̯̼͉̖͕̰͓͚͙̠̫̞̰̰̪͖̹̬̥̣̞̯̳̘̙͖̪̗̼̹̝̣͕̯̺̱͉̻̖͓͙̘̗̖̠̫̥̻̱̖͇̬͇̹͕͍̗̻̝̻͕̫̱̻͈̫͕̜̼͎̮̘̫̮̭͉̜͔͈̻͕͍̠̞͔̪̳͕̰͈͖͇͍͇͎͕̙̟͔͔͇̭̥̠͇̖͎͙̖̫͚͇͕̮̳̯̟̺̺̪̪̙̣̥̰̜̺̥̭̦̤̲͓̮̦̹͙̼͓̼̮̙̮̠͎͍͖̇͐͐͐̿́͊̃̀͋̔͑̓͂͌͗̋̇̎͛͊̋̔̓̇̚͜ͅč̷̨̡̡̨̫̗̠̥̫̩͕͉͉͇̮͉̲͎̭̝̬͚̜̮̼̰̭̞̘̠̘̰̹͈̯̫̟͓͙̻̤̰͈̌͂̓͐̑̌̏̊͂͂̉͌̐̇̎͋̍̉̑̃̇̃̎̓͋͛̑͊̾͊̔̍̄͂͘͘͝͝͠ǫ̴̡̧̨̧̧̢̢̧̧̢̨̢̨̡̧̨̡̡̨̨̨̡̡̨̡̨̨̨̧̧̡̛̛̤͚̖̫̰̣̣͍̱̜͈̻̙̲̙͚͚̖͕̠̼̲͚̯͚̳̻͇̘̲̦͕̦̜̣̙̣̣̜̰̝͕̤̝̫̺̳͙̮̬̪̹̲͍̣̹̙̠̫̘̥̦̘̰̞̙̟̟̤͓̙͖̣͓͔͓̩̩͈̗̤͇̠̞̩̮̪̥̪̱͚͍̝͕͎̞͔̩̖̲͔͈̩̻̩͕̫͓̳̙͓̞̟͙͉͉̬̮̗͓̱̲̮̯͇͕̰̖͚̦͈̞̺̠̳͕̭̭̳͓̻̯̞̳͔͍̟̬̳̩͚͎̲̹͍͇͈̙̳̞̖̗̯͖̱͖̯̤̠̲͍̩͈̭̙͈̲̱̲̼̩̘̘̜͔̲̱̯͔̈́̈́̋̿̋̿͋͋͌̏̽̇͗͊͑̔͆͌̿̋͌̋͗̉̊̿̐̄̈́̈́̈̄̆̅̃́̄̅̍́̽͒̈́̽̄̌̇̓̈͂͆̊͆̿́͗͛̅̋́͆͑͛̆̔̍̀̍̃̎͒̀͋͛̽́̏̄̓̌͘͘͘̚̕̕͘͜͜͜͜͜͜͠͠͠͝͠͝͝ͅͅͅd̷̨̧̧̢̢̧̛̛̛̛̛̛̛͖̭̘̺͕̜̬̤̭̬̠̫̤͍͚̪̬̣͉̳̮̱͖̻̟͓̫̹͚̝̗̳̰̺͍͎͉̟̱̜̫͇̫̯̼̠̞̝̤͖̖̻͍͖̰̻͕̙͙͚͈̱̝͉͙̘̰͚̩̗̟͕̞̞̼̣͖̜̳̥̼͉̘͈̘̩͕̦̺̝̟̼͍̥̲̤̪̗̀͌̊̃̆̑̓̓̌͌̏̎͋̀͌͐̈́̀̂̍̒̅̓̏̓̈́̀̀͊̒̎̓́̒̔̉̀̍̿̒͛̍̍̅̇͑̆͒̓̌̑̏̏̈͊̈́́͌̀̃̆́̔̃̀͛̾̅́̿̀́̿̒͆́̍̂̀̿̆͑̊̉͆́̒̑̅̽̂̄͂̏̿̍̽̃͂̈́̀͌̒͗̅̉̎̓̐̀̌̿̓̈̓͛̽̄̉̑̄̊͂̀̽̔̇̍̀͂̇̈͊͐͗̽͐͊͐̑͘͘͘͘̕̚̕̕̚̚̕͘͘͜͜͜͜͜͝͝͝͝͠͠͝͠͝͠͠͝͝͠ͅͅͅę̴̨̡̨̡̢̛͈̳̞͈͇͕͙̪̩̼̩̗̲̳̹̯̖̙̱͔̺̪͇̜̼̍͌̆̅̽͛̏̑͊͒͊͌́̇̄͊́̏́̄̈͆̿͌̌̀̈̃͊̈̀̽͒͑̊́̍̑̃̒̐́̓̈̃̀͛̽̔̎̀̄̑͌̾͒̊̀̓̆̀̕̕͘͜͜͠͝͠͝ͅͅ ̴̨̡̢̢̢̢̛̛̛̛̛̛̛̛̼̻̬̪͎̖̭̯̤̥̭͚̖̖͚̳͍͎̻̰̯̗̭͔͎͇̖̮̻̯̰̯̦̗͔̺͔̩͈̫̣̪͕̜͇͓͓̅͐́̃̿̀̍͂̽̂͒̇̉̿̑̀̈̒̇͋͗̓͑̒̿͒̃̏̏̔͐̓͐̽͛̆̈͗̿͆́͌́̀̇̈́̓̄̾̇̈́̀͑̽̔͒̌͑̓́̀̈́̀͊̓̏̾̿̓͒̅͋̓̂͆͊̎̾̆̀̾̏̆̈̿̆͗̀̿͊̒̌̓́͆̈́͂̍͆͗͌̇̇͋̅̍̈́̊̽̈́̑́̅͐́̌̉͆͊͑̓̿͆̊͒̑̑̉̑̔̀̀́̐̓̽͂͌̾͑̌͑̓͒̀͗̈́̑̀̋͂͆̓̍̆́͛̈́̈́̀́̀͋͂͛̎̑̌͊̅͑̔̓͛͂̓̇̾́͌̓́̆͋̓̓͑̔̈́̎̍͊̃̋̃͌͛̓́̔͆̈͐͒̂̅̂̓͂̋̅̽̏̉̎̊̈̿̾̊̃͆̆͊͂̎͋̌͊͌̍̄̔͒̄͗̈́͒̇̕̕̕̕̕̚͘̚̕͘̕̚͘͘͘̕͘̚͘̕̚͝͝͝͠͝͠͝͝͝͝͝͠͠͝͝͝͠͠͝͝ͅͅį̴̛͕͍̠̩͎͇̳̪̱̖̝͙͉̩̩̯̜͕͓͕̀͂͛̃͑̉͐̏͑́̒̃̑̐̾̈͐͝ͅs̶̨̢̨̡̧̡̡̧̨̪̱̪̙̤̥̺̰͚̦̞̫̟̭̟̠̗̺̲̺̹͙͇͈̭̱̪͔̦̦̻̭͙̱̱̬̙̺̤̤̙͈̭͖̯͇̞͙͈̟͓̖̠͚̳̤̺̙̤͔̯͍͔͖̱͈͍̞͚̗̭̮̣͕̻̝̮̯͐̑̃̌̐̈́̌̈̔́͋̂͊̿̈́̉̄̆̎̃̏͑̈̑̔̋͐̂̽̈́̔͗̒̌́̓̉̕̕͜͜͠ͅͅ ̴̢̡̧̧̡̢̡̨̢̢̧̡̧̨̡̡̨̧̧̡̨̛̛̛̱͚̭̯̯̘͍͕͓̱̯̩̪̠͓̫͕̖̠̤̱͕̬̞̘̭̗͍͙͚͎̗̫̘̹̫͔̹̱̟̻̬̞͙͇͉͔͙͍̟͙͈̪̞̤̪͉̫̠̤̫̭̦͍̰̪͎̠̲̣̰̠͍̪̦̞̬̘̟̳̣̼̜̻̬̗͎͓͓̳̙̳̩͙̼̬͍̝͓̲̰̤͇͚͖̠̹͖͓̜̳̳̼͈͈̝̘̹̪̱̳̱͎̙̳̩͕̞̻͍͓̗̪̖̣͚̤͇͈̳͓̝̗͔͇̖̲͙̤͉̺̮͔̞̫̱̮̻͇̼̯̹͓̥̪̩̹̳̰͍͓̖̟̮͉͔̰͙̲͓͇͉̞͓̥̖̗̘̜͖̱̯͎̺͓̬͎͕̘̻̻̥̲̖̬̯̰̞̜̫̬̪̲͎̠̳̥̫̜̠͍̼̟͓͈̻͍͈̙̮̠̱̻̫̼̯̜̯͓́͂͊̽͑̾̃̽̈́̒̓̒́̑̽͗̃̏̏̿̅̃͑̒́͌̈́́̒͊͊̆́͒̒̓͌̊͆̿̉̈́̇̑̃̇̋̾̒̽̎̍́̕̚̕̕̕̕͘̚̚͜͜͝͠͝͠͠͝ͅͅͅͅͅͅs̸̨̨̨̢̡̢̨̡̢̢̨̧̡̧̡̧̡̧̧̛̯͕̦̪̹̦͓͓̮̹͈̩͎̗̻͍̪̩̮͖̺͕͉̲̖̹̹̻͈̗͎̮̬͔̦̹͔̞̳̙̤͙͈̗̪̥̦͉̯̮͓̰̙̝͇̦̤̳̣̦͎̬̬͈͖̙̺͉̥̮̖͕̗̗͓̥͔̥̬̘͉̠̝͕̥̦̙͉͎͚͔̖͍͓̖̩̳͚͔̟̰̝̳͖̲̬̗̹͈͙̳̘̠̱͇͎̗̞̳̯̣͖͎͇̮̞̗̻̞̱̪̳͓̣̱͙̩̼͍͖̭͓͇̗̫͔̗̘̤͖͈̦̭̻͓̤͚͍̜̝̯͍͓͖̥̳̮͓̦͕̦͕̱͉̗͙̫̞͔̪͍̭͕̄̊͒͌͛̅̑̅̂͒̂̈́̃̂̀́̈͋̑̋̃̊̇̈̄̽̃̆̑̔͆̂̍͑͛͊̇̒̍̂̏̋̓̂́͐͆̎̿́̑̚̕͘̕͘͘͜͜͜͜͜͜͜͝͝͝͝͝͝ͅͅu̴̢̧̨̡̨̡̨̢̢̧̨̧̡̧̨̢̢̡̢̡̨̨̢̡̡̧̢̨͈͔̟̯͇̻͙̬̟͖̘͎͈̘͙̬̰͉̟̠͉̻̯͖̼̪͎͓͚̟̞̺͖̞͕͚̗͇͔̩̼̪͔̺̯̹͍̮̗͍͚̻̙̹͙͉͈̙͔̜̬̙̺̥̬̜̩̜̟̘̪͍̤̤̪͈͈͖̲̥͇̣͈̥͖̩̞̬̟̺̻̩̝͉̮̜̖͖̺͉̺̱̖̗̰͕͓̼̱̥̠̖̫̱̖̝̤̭̲̭̖̙͎̫̰͈̲͈̣̪̣̳͓̝͚̘̪̞͖̩̮̗̱͈͉̰̻̻̠̞͙̭̰̪͙̝̰̞͖̩͖͇̩̺̗̬̦͙͉̬̜̱̰̱͓̪͙̮̝̼̙̻͔͎̱͖͙͓̣̼̩̰̗͖̱̞̼͇̙̦̹̯̖͇̫͕͍̒̀̂̿͆̊̔̐̿̔̀͆͂̅̽̽̋̊̈́̈́̌͂̿̀̌́̔̉̑́̓̒̃̿͛̓̋̆͆́̈́̆̍̔͊͗̏̆̈́̑͑̓̀̆͘͘̕͘͘̚͜͜͜͜͝͝͠͝ͅͅͅͅͅc̶̡̢̢̢̢̨̡̨̢̧̢̨̢̨̧̢̢̡͍̖͎̪͉̼̮̲̣̪̘̮̯͖͖̼̯͙̻̮͍̲̖̙͕͖̯̠̪̯̲̞̞̠̳͈͚̜̟̙̫͎̫̱̩͈͚͎̮̱̝̼͚̺͚͇̪̱̫͇̱͈̟̲͇͔̝̯͎̗̣̘̘̺͈̼̦̖̺̖͉̬̫̥̲̣̞͔̣̣͚̤͇̻̫͉̥̖̦̫̪̠͈̙̰͈̤̤͎͕͎̙͔̪̭̼̞̙͇͕͎͔̼̘̖̦͚͔͉̫͕͔̜̮̱͉̠͓̪͕̼̳̖͙͍̭̬̞̻̬͔͕̑̐̄̆́͊͂͗̒̐̅̾͗̉̕͜͜͜͝ͅḫ̸̢̧̧̧̧̡̢̢̢̫̝̬̣̺̠̯̮͚̦̩͍̻̯̪̪̝̩̹̠̘̤͓͇̪͍̲̠͍̝͉̭̲̘̼͙͍̜̙̣̫̪̬͓̻̤͚͖͛̈̀̌͜͜͝ͅ ̴̨̢̡̡̨̨̡̢̧̡̨̡̡̛̛̛̛̹̭̗̖̹̰̼̗̳̹̯͔͚̻͚̙̹̰̪̺̩͈̳͉̼̗̝̳͖̞̯̠̭̯͎͓͎̘͉̺͇̬͇̯̜̯͓̳̞͚͍̭̯̦̺̳̘̰̲̲̜͓͔̼̺͍̟̠̙̱̞̲͉̣̮̭̗͈͕͚͚̣͓̻͍̩̣̻̲̳̹̲̫̮͚̲͍͎̰̮̮̯͖̰̥̝̮̞͍͇̹̹̫̞͔̫̭̥͉͉̱̯̻̥͈̑̔͊̆̌̇̇̌͆̍̓͛̅̂̊̂̃̌̋́͑̐̄̃̾̔͗̿͛̊̀̉͐̋͑̇̿̃̏̈́̏̔́̌̿̈́̃̈́́̈́̄̃͌̆̅̓̎̽́̑͊̑̈͛̌́̈́̿̿̐͐͗̾͛̉̐͊̏̉̏̉̏̌͋͊̍͒͐̑́̽̈́͊́̃̂͂̓̂͌̓̉̏͛̍̍̄̐̃͐̐͒̉̏̈̐̒̔̈́͒͆̈́̾̄͛̋̔̿̅́̃͌̎͐̀̿̊̍̿̊͌̚̚̚͘͘͘̚̕̕͜͜͜͜͠͠͠͠͝͝͝͝͝ͅͅf̶̨̢̡̢̢̢̢̡̨̡̢̧̢̨̨̧̢̨̨̨̨̢̧̢̢̦͇̼̹̫͔̬͔͎͇̱͉̜̤̟̩̖̯̞̱̗̺͎͖̙͖̱̻̩̮̯̲̝̥̟̠̰̘̮̖̹̲͖̖̪͖̲̖̫͈̞̫̣̗̖̝̹̟̙͓͙͖̺͉͚̪̣͓̮͉̦̪͕̗͈̩̗̤͈̮̱͈̙͉͙̭̼̺̻͙̟̥̬̤̤̺̼͚̣̗͙͇̬̭͇̖͇̖̣͎̜̭͙̠̤̫̫͙͕̺̱͙̟̤̦͓̦͍̜̖͉̥̤͓̹͉̮͇̙͍̙̠̳̲͉̦̮̣͇̠͕͙̫̗̲̘̞͎͕̗̹̞̟̺̥̤̥̪̝̹͎͚̦͍͓͓̘̟̼͙̯̠̮̖̲̲̪̜̖̝̙͉͚̼̳̘̘̻͕̘͓̜̯͙͕͉̬̲̲̼̥̭̰͕̣̻̬̫̬̼̖͈̺̞̹̘̭͈̼̣̮̘̜̗̦̬͓͇͉͍̮͕̲͙̗͍̝̝͉͔̻͔̭͇̟̞̜̘̗̥̲̉̎̀͛̈͊̋̾͋͑̾̃̽̽̓͆̉̃́̈́́͐͂͜͜͜͜͜͝͝͝ͅͅͅͅͅͅͅͅư̸̡̧̢̨̨̢̨̡̧̛̛̛͕̺̘̗͕̪̟̞̳͖̲̻̻̦̤̤͔̬͍̦͈̬͇̯̭͉̬͚̙͙̼͎̩̠̺͙̫̜͙̜̮̬͎̩̼̣͔̦̘̫̞͖͕̻̩̻̱̫̻͉̬͚̘̫̟̣̱̮̞͍̖̰̫͙̭̳͓̦̯͙̣̫̼͓̱͚̩̺͉̭̮̠̮̲̭̙̳͇͙͉̺̖͙̼͔͇̤͙͈̭̟̹̰͎̝͓̗̘̼̤̞̪̱̜̭̖̣͔̹̹̠̝̖̪͉͕̌̍̋́̈̾̓͆͗̑͆̄̓̏̃͒̄̎̐͌̍̐̓̎͊͂̉̍͆́͆̋̒̆̌͗̓͗͋̃̈́̈́̌̏͒̌̏͗̓̿͌̄̃̎̿͋͌̒̏̀͐͆̋̿̕̕̕͜͝͝͠͝͠͝͠͠ͅͅͅǹ̵̨̨̧̨̧̢̡̧̨̡̨̨̨̢̢̨̡̧̨̡̢̡̢̳̙̻̘͔͈̹͓̺̩̦̦̻̥͓͚̟͚͕͔͉̺͈̝͓̯͔͕͍͕̖̬̟̝̰̩̩̰͍̮̥̦̝̲̜̬̝̩͔̺͍͍̻̺̱͎̫͍̥͉̯̳͉͓̟͈̳̤͚̘̮̱̭̞̻̦̗̹̠͖̫̫̤̹̗̺͔̖̦̰̮̬̱̖̹͉̩̬̭̖̦͚̻̹̖̮͙͎̣͓̳̪̳͍͍̗̺̯͈̙͓̣̗̦͙̺͚̰̪͍̤̻̥̣̬̻̜̘̘͇̟̜̝̼̩͇̣̤̳̹̩̜͖̜̤̩̺̼̻̩̟̝̩̼̩̲͖̟̣̝͇̜͙̗̞̦̘͙̳͚̺̫̰̜̖͉̟̗͖͕̬̦̥̮̜̙̬̺͓̠̯̤̮̼̜͙͉̰̙̗͕͍͚̦͕̮͉͈̙̪͓̳̯̟̱̦̹̭̺̼͉̯̯͖̖̘̮̞̼͎̪͖͙̭͍̣̯͚̾̽͒͒̊̈̃̐͐͒̉̋̎͑͆̈̎͌̃̒͊̔͑̄̿͑̃̓͐͆̿̿̾̃̏̀̚̕͜͜͜͜͝͝ͅͅͅͅͅͅ!̶̡̢̨̨̨̡̡̨̡̨̨̨̢̢̧̢̢̢̢̡̨̛̛̛̛̭̲̺͔̰͓͓͈͍̖̮̭̤̩͚̭̩̼̫͈̹̙̭͚͇̗͖̙̙̼̰͔̭͓͎̯͓̯̜͚̗̝͔͉̼̠̹͚͇̩̬̬͓͚̭͎̠̖͖̬̞͉͎͎͉̘̩̬̺̳̖̻̘͎̹̹͕̟̗͉̰͖̮͔͇̘̞̭̮͈̲̪͉͈̻̹̻̣͖̠̙͎͍͉̤̭̳̞̳̝̠̳̺͈̺͍̮̭̺͚̹̫͓͎̱̹̰͓̺̘͓͇̙͙̱͉̟̙͖̭̙̳̰̮̻̬̖̲̹͖̝̬̳͉̗̫̮̮̹͚̹͕̤͓̺̮̜̜̻̩̦̪͓͎̬͍̗̺̝̗̣̘̖͇̬̻͍̮̜͚̱͍̗͔͎͙̗̳̞̩̩͇͕͈͙̬̻̮͕̤̲͚̘̥̙̻̲̠̦͍̩̺̩̭͓̘͙͎̳̝̞͚̄̍̇̎̊̉̃̄͐͆̒͗̈́͛̊͗̉̑̅̒̈́̀̈̈́̌̊̂̃̍̈́̊͒̂̆̎̈́͌́̆͑͆̈́̐͗͋̾̇̂̍̃͑̏͐̍̐̌̑̃̄̓̓̃̽̑̂̂͌͊͆͋̉̇̓̽͛̂̅̀̀̌̎̌̈́̐̽͊͒̀͛̏̌͐̈́̽͒̉̇͒͑͋͛́̽͋̋̊̋̒̈́͛̉̒͊́̓̋͂̋̓̅̃͗́́̾͋͛̃́́̋̎̌̓̄̽̔̌́̅̎̓̎͆́̋͊͆͒͒̈́̂̆͐͋̐̿͆̿̾̿̅͑́̿̏̌̅̌͊͐̔͌̽͆͒͋̈̇̀̈́́͑͐̍̏̀̓̀̓̐͑̓͒̉̂̇͐͐͌̀̈́̅̓͊̓͌̔͂̀͗̓͑́̈́̀̉̀̓̇͐̒͛̈́̀̉̏̐̃̈͂̋̋̀͛̈̌̎̏̐͛̑͊͗̐͘̚͘͘̚͘̚̕͘͘͘͘͘̚̕͘̕̕͘̚͜͜͜͜͜͜͜͜͜͝͠͠͠͝͠͝͝͠͝͝͝͠͝͝͠͝͠͝͝͝͠͝ͅͅͅͅͅͅ
    
    T̸h̴e̶ ̵p̷o̷w̶e̵r̷f̸u̷l̷ ̵p̴r̷o̷g̶r̷a̸m̸m̶i̸n̴g̴ ̷l̶a̴n̸g̵u̵a̶g̸e̶ ̸t̶h̴a̵t̵ ̶i̷s̶ ̵a̷l̴s̸o̷ ̵e̵a̷s̷y̴ ̵t̵o̷ ̷l̷e̶a̵r̴n̸.̵
    """#
  }
  return str
}()

let shortSample = #"""
    Swift 👨‍👨‍👧‍👧简c̴̭̈͘ǫ̷̯͋̊d̸͖̩̈̈́ḛ̴́🇺🇸🇨🇦🇺🇸코
    """#

let sampleString2 =
    #"""
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    The powerful programming language that is also easy to learn.
    Swift is a powerful and intuitive programming language for iOS, iPadOS, macOS, \#
    tvOS, and watchOS. Writing Swift code is interactive and fun, the syntax is \#
    concise yet expressive, and Swift includes modern features developers love. \#
    Swift code is safe by design and produces software that runs lightning-fast.
    
    """#
