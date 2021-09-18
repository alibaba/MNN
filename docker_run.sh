# using docker run test
docker start mnn_ci
docker exec -i -e TEST_ID=$(pwd | awk -F "/" '{print $(NF-1)}') mnn_ci bash <<'EOF'
cd ~/yanxing_zhaode/cise/space/$TEST_ID/source && ./test.sh linux
exit
EOF
