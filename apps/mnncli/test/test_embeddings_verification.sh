#!/bin/bash

# embeddings_bf16.bin 文件校验测试脚本
# 用于重现SHA1/SHA256校验过程

set -e

echo "=== embeddings_bf16.bin 文件校验测试 ==="
echo

# 配置
FILE_URL="https://huggingface.co/taobao-mnn/SmolVLM-256M-Instruct-MNN/resolve/main/embeddings_bf16.bin"
TEMP_FILE="/tmp/test_embeddings_bf16.bin"
PYTHON_SCRIPT="/tmp/verify_hash.py"

echo "1. 获取文件元数据..."
echo "URL: $FILE_URL"
echo

# 获取HEAD请求，查看重定向和ETag
echo "发送HEAD请求..."
HEAD_RESPONSE=$(curl -s -I "$FILE_URL")
echo "HEAD响应状态:"
echo "$HEAD_RESPONSE" | head -10
echo

# 从原始响应中提取x-linked-etag
EXPECTED_ETAG=$(echo "$HEAD_RESPONSE" | grep -i "x-linked-etag:" | cut -d' ' -f2- | tr -d '\r' | sed 's/^"//' | sed 's/"$//')
if [ -z "$EXPECTED_ETAG" ]; then
    echo "❌ 未找到x-linked-etag!"
    exit 1
fi
echo "提取的x-linked-etag: $EXPECTED_ETAG"
echo "ETag长度: ${#EXPECTED_ETAG}"
echo

# 提取重定向URL
REDIRECT_URL=$(echo "$HEAD_RESPONSE" | grep -i "location:" | cut -d' ' -f2- | tr -d '\r')
if [ -n "$REDIRECT_URL" ]; then
    echo "重定向URL: $REDIRECT_URL"
    echo
    
    # 获取最终URL的HEAD信息
    echo "获取最终URL的HEAD信息..."
    FINAL_HEAD=$(curl -s -I "$REDIRECT_URL")
    echo "最终HEAD响应:"
    echo "$FINAL_HEAD" | head -10
    echo
else
    echo "❌ 未找到重定向URL!"
    exit 1
fi

echo "2. 下载文件..."
echo "开始下载到: $TEMP_FILE"
# 使用-L参数跟随重定向
curl -s -L "$FILE_URL" -o "$TEMP_FILE"

if [ ! -f "$TEMP_FILE" ]; then
    echo "❌ 下载失败!"
    exit 1
fi

FILE_SIZE=$(wc -c < "$TEMP_FILE")
echo "下载完成! 文件大小: $FILE_SIZE bytes ($(($FILE_SIZE / 1024 / 1024)) MB)"
echo

echo "3. 验证文件哈希..."
echo "预期ETag: $EXPECTED_ETAG"
echo "ETag长度: ${#EXPECTED_ETAG}"

# 创建Python验证脚本
cat > "$PYTHON_SCRIPT" << 'EOF'
import hashlib
import sys

def verify_file_hash(file_path, expected_etag):
    print(f"验证文件: {file_path}")
    
    # 读取文件
    with open(file_path, 'rb') as f:
        data = f.read()
    
    print(f"文件大小: {len(data)} bytes ({len(data)/1024/1024:.1f} MB)")
    print(f"预期ETag: {expected_etag}")
    print(f"ETag长度: {len(expected_etag)}")
    
    # 根据ETag长度判断算法
    if len(expected_etag) == 40:
        print("使用Git SHA1算法")
        # 计算Git SHA1
        sha1 = hashlib.sha1()
        sha1.update(b'blob ')
        sha1.update(str(len(data)).encode())
        sha1.update(b'\0')
        sha1.update(data)
        calculated_hash = sha1.hexdigest()
        algorithm = "Git SHA1"
    elif len(expected_etag) == 64:
        print("使用SHA256算法")
        # 计算SHA256
        sha256 = hashlib.sha256()
        sha256.update(data)
        calculated_hash = sha256.hexdigest()
        algorithm = "SHA256"
    else:
        print(f"错误: 意外的ETag长度 {len(expected_etag)}")
        return False
    
    print(f"计算的{algorithm}: {calculated_hash}")
    print(f"匹配结果: {calculated_hash == expected_etag}")
    
    return calculated_hash == expected_etag

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python3 script.py <file_path> <expected_etag>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    expected_etag = sys.argv[2]
    
    success = verify_file_hash(file_path, expected_etag)
    sys.exit(0 if success else 1)
EOF

# 运行Python验证脚本
python3 "$PYTHON_SCRIPT" "$TEMP_FILE" "$EXPECTED_ETAG"
VERIFICATION_RESULT=$?

echo
if [ $VERIFICATION_RESULT -eq 0 ]; then
    echo "✅ 校验通过!"
else
    echo "❌ 校验失败!"
fi

echo
echo "4. 模拟C++代码处理过程..."
echo "模拟ETag处理:"

# 模拟C++代码的ETag处理
ETAG_WITH_QUOTES="\"$EXPECTED_ETAG\""
echo "原始ETag: $ETAG_WITH_QUOTES"

# toLower
ETAG_LOWER=$(echo "$ETAG_WITH_QUOTES" | tr '[:upper:]' '[:lower:]')
echo "toLower后: $ETAG_LOWER"

# 去除引号
ETAG_CLEAN=$(echo "$ETAG_LOWER" | sed 's/^"//' | sed 's/"$//')
echo "去除引号后: $ETAG_CLEAN"
echo "ETag长度: ${#ETAG_CLEAN}"

# 判断算法
if [ ${#ETAG_CLEAN} -eq 40 ]; then
    echo "将使用Git SHA1算法"
elif [ ${#ETAG_CLEAN} -eq 64 ]; then
    echo "将使用SHA256算法"
else
    echo "错误: 意外的ETag长度 ${#ETAG_CLEAN}"
fi

echo
echo "5. 清理临时文件..."
rm -f "$TEMP_FILE" "$PYTHON_SCRIPT"
echo "清理完成!"

echo
echo "=== 测试完成 ==="
echo "如果所有步骤都显示通过，说明curl下载和校验逻辑是正确的。"
echo "C++代码的问题可能在于："
echo "1. 下载不完整"
echo "2. 文件路径错误" 
echo "3. OpenSSL编译问题"
echo "4. 缺少调试日志"
