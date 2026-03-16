#!/usr/bin/env node

/**
 * OSS upload utilities for MnnLlmChat CDN release.
 * Reference: oneday-code-feature/skills_worktree/scripts/lib/oss-utils.mjs
 *
 * Environment variables (compatible with release.config):
 *   CDN_ENDPOINT   - e.g. https://oss-cn-hangzhou.aliyuncs.com
 *   CDN_ACCESS_KEY - Aliyun access key ID
 *   CDN_SECRET_KEY - Aliyun access key secret
 *   CDN_BUCKET     - OSS bucket name
 *
 * Alternative (OneDay-style):
 *   OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_REGION, OSS_BUCKET
 */

import OSS from "ali-oss"

function getOssConfig() {
	const accessKeyId =
		process.env.OSS_ACCESS_KEY_ID || process.env.CDN_ACCESS_KEY
	const accessKeySecret =
		process.env.OSS_ACCESS_KEY_SECRET || process.env.CDN_SECRET_KEY
	const bucket = process.env.OSS_BUCKET || process.env.CDN_BUCKET
	const endpoint = process.env.CDN_ENDPOINT || process.env.OSS_ENDPOINT
	const region =
		process.env.OSS_REGION || extractRegionFromEndpoint(endpoint)

	return { accessKeyId, accessKeySecret, bucket, region }
}

/**
 * Extract OSS region from endpoint (e.g. https://oss-cn-hangzhou.aliyuncs.com -> oss-cn-hangzhou)
 */
function extractRegionFromEndpoint(endpoint) {
	if (!endpoint || typeof endpoint !== "string") return null
	const m = endpoint.match(/oss-([a-z0-9-]+)\.aliyuncs\.com/i)
	return m ? `oss-${m[1]}` : null
}

export function validateEnv() {
	const { accessKeyId, accessKeySecret, bucket, region } = getOssConfig()
	const missing = []
	if (!accessKeyId) missing.push("CDN_ACCESS_KEY or OSS_ACCESS_KEY_ID")
	if (!accessKeySecret) missing.push("CDN_SECRET_KEY or OSS_ACCESS_KEY_SECRET")
	if (!bucket) missing.push("CDN_BUCKET or OSS_BUCKET")
	if (!region) missing.push("CDN_ENDPOINT or OSS_REGION")
	if (missing.length > 0) {
		console.error("❌ Missing OSS env vars:", missing.join(", "))
		console.error("   Add CDN_ACCESS_KEY, CDN_SECRET_KEY, CDN_BUCKET, CDN_ENDPOINT to ~/.zshrc")
		process.exit(1)
	}
}

export function createOssClient() {
	const { accessKeyId, accessKeySecret, bucket, region } = getOssConfig()
	return new OSS({
		region,
		accessKeyId,
		accessKeySecret,
		bucket,
		authorizationV4: true,
	})
}

/**
 * Upload a local file to OSS.
 * @param {string} ossPath - OSS object path (e.g. releases/0.8.1.3/mnn_chat_0_8_1_3.apk)
 * @param {string} localFile - Local file path
 * @returns {Promise<{url: string}>}
 */
export async function uploadFile(ossPath, localFile) {
	const client = createOssClient()
	console.log(`⬆️  Uploading ${localFile} -> oss://${getOssConfig().bucket}/${ossPath}`)
	const result = await client.put(ossPath, localFile)
	console.log(`✅ Uploaded: ${result.url}`)
	return result
}
