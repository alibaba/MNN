#!/usr/bin/env node

/**
 * Upload model_market.json to OSS (update market config).
 *
 * Usage:
 *   node scripts/upload-market-config.mjs
 *   node scripts/upload-market-config.mjs --dev
 *   node scripts/upload-market-config.mjs --file app/src/main/assets/model_market.json
 *
 * OSS path: data/mnn/apis/model_market.json (or model_market_dev.json with --dev)
 * Env: CDN_ACCESS_KEY, CDN_SECRET_KEY, CDN_BUCKET, CDN_ENDPOINT (~/.zshrc or release.config)
 */

import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { uploadFile, validateEnv } from "./oss-utils.mjs"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, "..")

const DEFAULT_LOCAL = path.join(ROOT, "app", "src", "main", "assets", "model_market.json")

function parseArgs() {
	const args = process.argv.slice(2)
	let localFile = DEFAULT_LOCAL
	let ossPath = "data/mnn/apis/model_market.json"

	for (let i = 0; i < args.length; i++) {
		if (args[i] === "--file" && args[i + 1]) {
			localFile = path.resolve(ROOT, args[i + 1])
			i++
		} else if (args[i] === "--dev") {
			ossPath = "data/mnn/apis/model_market_dev.json"
		}
	}
	return { localFile, ossPath }
}

async function main() {
	const { localFile, ossPath } = parseArgs()

	if (!fs.existsSync(localFile)) {
		console.error("❌ File not found:", localFile)
		process.exit(1)
	}

	console.log("📤 Update market config")
	console.log(`   Local: ${localFile}`)
	console.log(`   OSS:   ${ossPath}`)
	console.log("")

	validateEnv()
	await uploadFile(ossPath, localFile)

	console.log("")
	console.log("✅ Market config updated")
}

main().catch((err) => {
	console.error("❌ Upload failed:", err.message)
	process.exit(1)
})
