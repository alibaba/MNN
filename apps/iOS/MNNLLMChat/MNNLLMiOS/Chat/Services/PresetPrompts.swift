//
//  PresetPrompts.swift
//  MNNLLMiOS
//

import Foundation

struct PresetPrompt {
    let title: String
    let icon: String
    let text: String
    let imageBundlePath: String?
    let audioBundlePath: String?
    let audioSHA256: String?

    init(
        title: String,
        icon: String,
        text: String,
        imageBundlePath: String?,
        audioBundlePath: String? = nil,
        audioSHA256: String? = nil
    ) {
        self.title = title
        self.icon = icon
        self.text = text
        self.imageBundlePath = imageBundlePath
        self.audioBundlePath = audioBundlePath
        self.audioSHA256 = audioSHA256
    }

    var isMultimodal: Bool { imageBundlePath != nil }
    var isASRAudio: Bool { audioBundlePath != nil }
}

enum PresetPrompts {
    private static let pocketModelName = "MNN-Pocket-0.3B"
    private static let qwenMopdModelNames = ["Qwen3.5-2B-mopd-dflash", "Qwen3.5-2B-mopd"]

    private static let qwenMopd512Prompt = """
    请严格按 JSON 格式输出，不要输出任何多余文字。

    从订单详情文本抽取为 JSON，字段为 product、quantity、price：

    订单编号：SO-20260315-8842，下单时间：2026年3月15日14时22分，订单状态：已发货，预计3月18日前送达。
    收货信息：收货人陈女士，联系电话138****6721，收货地址为杭州市余杭区文一西路x号x座12层。
    商品明细：客户购买了3台笔记本电脑，型号为"锐思Book Pro 14"，配置为16GB内存、512GB固态硬盘，单价每台6500元，合计19500元。商家赠送电脑包3个、无线鼠标3个。发票与售后：开具增值税专用发票，发票抬头为杭州锐创科技有限公司；整机保修两年，电池保修一年，支持七天无理由退货，退货运费由买家承担。物流信息：承运商为顺达速运，运单号SF88123456789，3月16日上午已揽收，可在官网实时查询轨迹。历史订单：该客户曾于2025年11月采购同型号电脑2台，累计采购金额32500元，系统已自动将其升级为企业大客户。付款信息：本单采用对公转账方式支付，款项已于3月16日到账，财务系统同步生成了电子回单。交付要求：买家备注请安排工作日上午送达，签收时需出示单位介绍信，收货人当场开箱验机并签署验收单。增值服务：商家为本批设备免费预装正版操作系统与办公软件，并提供两年期每年两次的上门巡检服务。补充说明：本单属于企业年度办公设备更新计划的一部分，后续批次预计于六月底前完成采购，届时将沿用相同的价格协议与服务条款；商家承诺如遇同型号产品降价，未发货部分可按新价格重新核算，已发货部分不再追溯调整，相关约定已写入双方签订的框架采购合同附件。验收标准：到货后由买家信息中心会同行政部门共同验收，验收内容包括外观完好性、配置一致性、序列号与合同清单比对以及开机自检，全部通过后方可在验收单上盖章确认；验收过程中如发现任何批次性质量问题，买家有权整批拒收并要求商家在三个工作日内补发新机。
    """

    private static let qwenMopdCodePrompt = "用 Python 写一个二分查找函数，在升序列表中查找目标值，返回索引，找不到返回 -1。"

    static func all(for model: ModelInfo) -> [PresetPrompt] {
        var presets: [PresetPrompt] = []
        let modelAliases = identityAliases(for: model)

        if isPocketModel(modelAliases) {
            presets.append(contentsOf: pocketPresets())
        } else {
            if isQwenMopdModel(modelAliases) {
                presets.append(PresetPrompt(
                    title: "512 tokens",
                    icon: "doc.text",
                    text: qwenMopd512Prompt,
                    imageBundlePath: nil
                ))
            } else if let text512 = loadText("512.txt") {
                presets.append(PresetPrompt(title: "512 tokens", icon: "doc.text", text: text512, imageBundlePath: nil))
            }
            if let text1024 = loadText("1024.txt") {
                presets.append(PresetPrompt(title: "1024 tokens", icon: "doc.text.fill", text: text1024, imageBundlePath: nil))
            }

            presets.append(imagePreset(
                file: "cat.jpg",
                title: NSLocalizedString("preset.cat", value: "图片·小猫", comment: ""),
                text: NSLocalizedString(
                    "preset.catPrompt",
                    value: "请仔细观察这张图片，然后用完整的几句话描述：画面的主体是什么、它的颜色和姿态、所处的背景环境，以及这张照片整体的氛围。",
                    comment: ""
                )
            ))
            presets.append(imagePreset(
                file: "shapes.jpg",
                title: NSLocalizedString("preset.shapes", value: "图片·形状", comment: ""),
                text: NSLocalizedString(
                    "preset.shapesPrompt",
                    value: "请仔细观察这张图片，逐一说明图中每个几何图形的形状、颜色和大致位置，最后总结一共有几个图形。",
                    comment: ""
                )
            ))

            if let audio = audioPreset(
                file: "audio (1).wav",
                title: NSLocalizedString("preset.audio", value: "音频·浙江地级市", comment: "")
            ) {
                presets.append(audio)
            }
        }

        let codePrompt = isQwenMopdModel(modelAliases)
            ? qwenMopdCodePrompt
            : NSLocalizedString(
                "preset.codePrompt",
                value: "请用C++实现一个冒泡排序，直接给出代码",
                comment: ""
            )
        presets.append(PresetPrompt(
            title: NSLocalizedString("preset.code", value: "代码题", comment: ""),
            icon: "chevron.left.forwardslash.chevron.right",
            text: codePrompt,
            imageBundlePath: nil
        ))
        return presets
    }

    private static func identityAliases(for model: ModelInfo) -> [String] {
        var aliases = [model.name, model.modelName]
        if let localSource = model.sources?["local"] {
            aliases.append(URL(fileURLWithPath: localSource).lastPathComponent)
        }
        return aliases
    }

    private static func isPocketModel(_ aliases: [String]) -> Bool {
        aliases.contains {
            $0.caseInsensitiveCompare(pocketModelName) == .orderedSame ||
                $0.caseInsensitiveCompare("slm270M") == .orderedSame
        }
    }

    private static func isQwenMopdModel(_ aliases: [String]) -> Bool {
        aliases.contains { alias in
            qwenMopdModelNames.contains {
                alias.caseInsensitiveCompare($0) == .orderedSame
            } || alias.caseInsensitiveCompare("Qwen3.5-2B") == .orderedSame
        }
    }

    private static func pocketPresets() -> [PresetPrompt] {
        [
            PresetPrompt(
                title: "数学题",
                icon: "function",
                text: "4个盒子里各有6个球，盒子外还有2个球，一共有多少个球？",
                imageBundlePath: nil
            ),
            PresetPrompt(
                title: "信息提取",
                icon: "text.magnifyingglass",
                text: "请从这句话中提取收件人姓名和地址：‘请把快递寄给李明，地址是北京市朝阳区建国路88号。’",
                imageBundlePath: nil
            ),
        ]
    }

    private static func imagePreset(file: String, title: String, text: String) -> PresetPrompt {
        PresetPrompt(title: title, icon: "photo", text: text, imageBundlePath: bundlePath("LocalModel/preset_images/\(file)"))
    }

    private static func audioPreset(file: String, title: String) -> PresetPrompt? {
        guard let path = bundlePath("LocalModel/preset_audio/\(file)") else { return nil }
        return PresetPrompt(
            title: title,
            icon: "waveform",
            text: "",
            imageBundlePath: nil,
            audioBundlePath: path,
            audioSHA256: "b112c5b9b4503e59b14102be18bba6cc0cb9cd44a86af87c2b75a31e1f06f066"
        )
    }

    private static func bundlePath(_ relative: String) -> String? {
        guard let root = Bundle.main.resourcePath else { return nil }
        let path = (root as NSString).appendingPathComponent(relative)
        return FileManager.default.fileExists(atPath: path) ? path : nil
    }

    private static func loadText(_ file: String) -> String? {
        guard let path = bundlePath("LocalModel/\(file)") else { return nil }
        let text = try? String(contentsOfFile: path, encoding: .utf8)
        guard let text, !text.isEmpty else { return nil }
        return text
    }
}
