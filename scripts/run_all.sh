#!/bin/bash

# CRLOT-DSP Phase 0 전체 벤치마크 실행 스크립트
# 목표: P0 블로킹 이슈 해결 및 품질 게이트 확인

set -euo pipefail

# 환경 설정
export TZ=Asia/Seoul
TIMESTAMP=$(date '+%Y-%m-%d_%H%M%S_KST')
OUT_DIR="out"
RELEASE_FLAGS="-c opt --copt=-O3 --copt=-DNDEBUG --copt=-march=native"

# 출력 디렉토리 생성
mkdir -p "${OUT_DIR}/e2e_benchmark"
mkdir -p "${OUT_DIR}/performance_benchmark"
mkdir -p "${OUT_DIR}/logs"

echo "=== CRLOT-DSP Phase 0 벤치마크 시작 ==="
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "출력 디렉토리: ${OUT_DIR}"

# 1. Release 빌드
echo "1. Release 빌드 중..."
bazel build ${RELEASE_FLAGS} //bench:e2e_benchmark //bench:performance_benchmark

# 2. E2E 벤치마크 실행
echo "2. E2E 벤치마크 실행 중..."
E2E_JSON="${OUT_DIR}/e2e_benchmark/e2e_${TIMESTAMP}.json"
E2E_LOG="${OUT_DIR}/e2e_benchmark/e2e_${TIMESTAMP}.log"

bazel run ${RELEASE_FLAGS} //bench:e2e_benchmark -- \
    --benchmark_format=json \
    --benchmark_out="${E2E_JSON}" \
    --benchmark_repetitions=3 \
    --benchmark_display_aggregates_only=false 2>&1 | tee "${E2E_LOG}"

# 3. Performance 벤치마크 실행 (SIGFPE 체크)
echo "3. Performance 벤치마크 실행 중..."
PERF_JSON="${OUT_DIR}/performance_benchmark/perf_${TIMESTAMP}.json"
PERF_LOG="${OUT_DIR}/performance_benchmark/perf_${TIMESTAMP}.log"

bazel run ${RELEASE_FLAGS} //bench:performance_benchmark -- \
    --benchmark_format=json \
    --benchmark_out="${PERF_JSON}" \
    --benchmark_repetitions=100 \
    --benchmark_display_aggregates_only=false 2>&1 | tee "${PERF_LOG}"

# 4. 전체 테스트 실행
echo "4. 전체 테스트 실행 중..."
TEST_LOG="${OUT_DIR}/logs/tests_${TIMESTAMP}.log"
bazel test ${RELEASE_FLAGS} --test_output=all ... 2>&1 | tee "${TEST_LOG}"

# 5. 환경 정보 수집
echo "5. 환경 정보 수집 중..."
ENV_LOG="${OUT_DIR}/logs/env_${TIMESTAMP}.log"
{
    echo "=== Build Configuration ==="
    echo "Timestamp: ${TIMESTAMP}"
    echo "Release Flags: ${RELEASE_FLAGS}"
    echo "Bazel Version: $(bazel version)"
    echo ""
    echo "=== System Information ==="
    echo "OS: $(uname -a)"
    echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')"
    echo "Memory: $(system_profiler SPHardwareDataType | grep Memory 2>/dev/null || echo 'Unknown')"
    echo ""
    echo "=== Git Information ==="
    echo "Commit: $(git rev-parse HEAD)"
    echo "Branch: $(git branch --show-current)"
    echo "Status: $(git status --porcelain)"
} > "${ENV_LOG}"

echo "=== CRLOT-DSP Phase 0 벤치마크 완료 ==="
echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "결과 위치: ${OUT_DIR}"

echo ""
echo "주요 결과 파일:"
echo "- E2E JSON: ${E2E_JSON}"
echo "- E2E Log: ${E2E_LOG}"
echo "- Performance JSON: ${PERF_JSON}"
echo "- Performance Log: ${PERF_LOG}"
echo "- Test Log: ${TEST_LOG}"
echo "- Environment Log: ${ENV_LOG}"
