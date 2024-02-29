# using docker run release
docker start mnn_release
docker exec -i -e TEST_ID=$(pwd | awk -F "/" '{print $(NF-1)}') mnn_release bash <<'EOF'
cd ~/yanxing_zhaode/cise/space/$TEST_ID/source && ./release.sh pymnn
exit
EOF