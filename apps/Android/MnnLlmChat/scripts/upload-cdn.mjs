#!/usr/bin/env node

/**
 * MnnLlmChat CDN upload script (ali-oss SDK, no ossutil required).
 * Reference: oneday-code-feature/skills_worktree/scripts/publish-vscode.mjs
 *
 * Usage:
 *   node scripts/upload-cdn.mjs
 *   node scripts/upload-cdn.mjs --apk release_outputs/cdn/mnn_chat_0_8_1_3.apk --version 0.8.1.3
 *
 * Environment: source scripts/release.config or set CDN_* / OSS_* env vars.
 */

import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { uploadFile, validateEnv } from "./oss-utils.mjs"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, "..")

const CDN_UPLOAD_DIR = path.join(ROOT, "release_outputs", "cdn")
// OSS path prefix, e.g. data/mnn/apks -> https://bucket.oss-cn-xxx.aliyuncs.com/data/mnn/apks/mnn_chat_0_8_1_3.apk
const OSS_APK_PREFIX = process.env.CDN_OSS_PREFIX || process.env.OSS_APK_PREFIX || "data/mnn/apks"

function parseArgs() {
	const args = process.argv.slice(2)
	let apkPath = null
	let version = null
	for (let i = 0; i < args.length; i++) {
		if (args[i] === "--apk" && args[i + 1]) {
			apkPath = path.resolve(ROOT, args[i + 1])
			i++
		} else if (args[i] === "--version" && args[i + 1]) {
			version = args[i + 1]
			i++
		}
	}
	return { apkPath, version }
}

function getVersionFromGradle() {
	const gradlePath = path.join(ROOT, "app", "build.gradle")
	if (!fs.existsSync(gradlePath)) return null
	const content = fs.readFileSync(gradlePath, "utf-8")
	const m = content.match(/versionName\s+"([^"]+)"/)
	return m ? m[1] : null
}

function findLatestApk(dir) {
	if (!fs.existsSync(dir)) return null
	const files = fs.readdirSync(dir).filter((f) => f.endsWith(".apk"))
	if (files.length === 0) return null
	const sorted = files.sort((a, b) => {
		const statA = fs.statSync(path.join(dir, a))
		const statB = fs.statSync(path.join(dir, b))
		return statB.mtimeMs - statA.mtimeMs
	})
	return path.join(dir, sorted[0])
}

async function main() {
	const { apkPath, version: versionArg } = parseArgs()

	const apk = apkPath || findLatestApk(CDN_UPLOAD_DIR)
	if (!apk || !fs.existsSync(apk)) {
		console.error("❌ APK not found. Run release.sh first or pass --apk <path>")
		process.exit(1)
	}

	const version = versionArg || getVersionFromGradle()
	if (!version) {
		console.error("❌ Version not found. Pass --version <x.y.z> or ensure app/build.gradle has versionName")
		process.exit(1)
	}

	const versionFilename = version.replace(/\./g, "_")
	const apkFilename = `mnn_chat_${versionFilename}.apk`
	const ossPath = `${OSS_APK_PREFIX}/${apkFilename}`

	console.log("📦 MnnLlmChat CDN upload")
	console.log(`   APK: ${apk}`)
	console.log(`   Version: ${version}`)
	console.log(`   OSS path: ${ossPath}`)
	console.log("")

	validateEnv()
	await uploadFile(ossPath, apk)

	console.log("")
	console.log("✅ CDN upload completed")
}

main().catch((err) => {
	console.error("❌ Upload failed:", err.message)
	process.exit(1)
})
