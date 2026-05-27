#!/usr/bin/env bash
# Launch the v3 CONUS scan on a spot GPU instance.
# Same instance constraints as training: g4dn.8xlarge primary, g6.8xlarge fallback.
#
# Env:
#   V3_SCAN_SAMPLE_N=10000  — validation sample; omit/empty for full scan
set -uo pipefail
cd "$(dirname "$0")/../../.."

[ -f sites_us/.env.agent-profile ] && set -a && . sites_us/.env.agent-profile && set +a
[ -f sites_us/.env ] && set -a && . sites_us/.env && set +a

REGION="${AWS_REGION:-us-west-2}"
BUCKET="${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set}"
AMI="ami-0b1c3e116f347cc8a"
KEY="industrials-scanner-key"
SG="sg-0e344ef6a61de3d56"
PROFILE="industrials-scanner-profile"
SUBNETS=(subnet-096bbbe80d8ea7409 subnet-0463b6600119e939d subnet-0c177d1da88790caa subnet-0e7a01003e03380f8)
TYPES=(g4dn.4xlarge g4dn.8xlarge)

V3_SCAN_SAMPLE_N="${V3_SCAN_SAMPLE_N:-}"
V3_SCAN_MAX_CHUNKS="${V3_SCAN_MAX_CHUNKS:-0}"
V3_SCAN_ONLY_CHUNKS="${V3_SCAN_ONLY_CHUNKS:-}"
V3_NUM_WORKERS="${V3_NUM_WORKERS:-}"
V3_BATCH_SIZE="${V3_BATCH_SIZE:-}"
V3_PREFETCH_FACTOR="${V3_PREFETCH_FACTOR:-}"
TAG_SUFFIX="full"
[ -n "$V3_SCAN_SAMPLE_N" ] && TAG_SUFFIX="sample-${V3_SCAN_SAMPLE_N}"
[ "$V3_SCAN_MAX_CHUNKS" != "0" ] && TAG_SUFFIX="diag-${V3_SCAN_MAX_CHUNKS}c"
[ -n "$V3_SCAN_ONLY_CHUNKS" ] && TAG_SUFFIX="probe-$(echo "$V3_SCAN_ONLY_CHUNKS" | tr ',' '-')"

UD="$(mktemp)"
cat > "$UD" <<EOF
#!/bin/bash
exec > /var/log/user-data.log 2>&1
set -x
sudo -u ubuntu -H bash -c '
  cd /home/ubuntu
  aws s3 cp s3://${BUCKET}/v3-scan-bundle/bootstrap_v3_scan.sh ./bootstrap_v3_scan.sh
  aws s3 cp s3://${BUCKET}/v3-scan-bundle/.env ./.env
  BUCKET=${BUCKET} \
    V3_SCAN_SAMPLE_N=${V3_SCAN_SAMPLE_N} \
    V3_SCAN_MAX_CHUNKS=${V3_SCAN_MAX_CHUNKS} \
    V3_SCAN_ONLY_CHUNKS=${V3_SCAN_ONLY_CHUNKS} \
    V3_NUM_WORKERS=${V3_NUM_WORKERS} \
    V3_BATCH_SIZE=${V3_BATCH_SIZE} \
    V3_PREFETCH_FACTOR=${V3_PREFETCH_FACTOR} \
    setsid nohup bash bootstrap_v3_scan.sh > /home/ubuntu/v3_scan.out 2>&1 < /dev/null &
'
EOF

try_launch() {
  aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$1" \
    --key-name "$KEY" --security-group-ids "$SG" --subnet-id "$2" \
    --iam-instance-profile "Name=$PROFILE" \
    --instance-initiated-shutdown-behavior terminate \
    --instance-market-options 'MarketType=spot' \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}' \
    --user-data "file://$UD" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=v3-scan-${TAG_SUFFIX}}]" \
    --query 'Instances[0].InstanceId' --output text 2>&1
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  for itype in "${TYPES[@]}"; do
    for subnet in "${SUBNETS[@]}"; do
      echo "[launch-scan] attempt $attempt: $itype / $subnet (spot, ${TAG_SUFFIX})"
      out="$(try_launch "$itype" "$subnet")"
      if [[ "$out" == i-* ]]; then
        echo "[launch-scan] SUCCESS: $itype $out"
        echo "$out" > /tmp/v3_scan_instance_id.txt
        rm -f "$UD"
        exit 0
      fi
      echo "[launch-scan]   declined: $(echo "$out" | tr -d '\n' | cut -c1-220)"
    done
  done
  echo "[launch-scan] no spot any type/AZ — waiting 300s (NEVER on-demand)"
  sleep 300
done
