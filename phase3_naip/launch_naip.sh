#!/usr/bin/env bash
# Launch the Phase 3 NAIP worker on a SPOT GPU instance.
#
# SPOT ONLY — never on-demand (per [[never_on_demand]]). g7e.2xlarge primary
# (Blackwell RTX PRO 6000, 96 GB VRAM — most cost-effective for GPU-bound SAM 3),
# g6e.2xlarge fallback (L40S, 48 GB VRAM). Phase 3 NAIP is GPU-bound: g6.8xlarge
# benchmark showed CPU 3%, RAM 3%, GPU 68%, so a beefier GPU on a smaller box
# dominates. SPS (us-west-2 2026-05-22): g7e.2xlarge=3 in usw2-az3, =1 elsewhere
# — so the launcher will mostly land in az3 or fall back to g6e.2xlarge (clean
# 3 across all 4 AZs). Spot quota 32 vCPU; both types are 8 vCPU, well under.
# Tries every default-VPC AZ in turn; on capacity-fail across all AZs, waits
# 5 min and retries the sweep.
#
# The instance auto-runs bootstrap_naip.sh via user-data and self-terminates
# on completion or failure (bootstrap EXIT trap + instance-role TerminateInstances).
#
# Env knobs forwarded to bootstrap:
#   SHARD  — N/M, default 0/1 (single-instance whole-dataset run)
#   LIMIT  — cap on clusters this run (for benchmarks: SHARD=0/1 LIMIT=20)
set -uo pipefail
cd "$(dirname "$0")/../.."
[ -f sites_us/.env ] && set -a && . ./sites_us/.env && set +a

REGION="${AWS_REGION:-us-west-2}"
BUCKET="${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set}"
AMI="ami-0b1c3e116f347cc8a"          # DLAMI OSS Nvidia PyTorch 2.6, Ubuntu 22.04
KEY="industrials-scanner-key"
SG="sg-0e344ef6a61de3d56"
PROFILE="industrials-scanner-profile"
SUBNETS=(subnet-096bbbe80d8ea7409 subnet-0463b6600119e939d subnet-0c177d1da88790caa subnet-0e7a01003e03380f8)
TYPES=(g7e.2xlarge g6e.2xlarge)

SHARD="${SHARD:-0/1}"
LIMIT="${LIMIT:-}"

UD="$(mktemp)"
cat > "$UD" <<EOF
#!/bin/bash
exec > /var/log/user-data.log 2>&1
set -x
# Run the bootstrap as ubuntu. Use 'sudo -u ubuntu -H', NOT 'sudo -iu ubuntu':
# the -i (login) flag re-quotes through the login shell and collapses the
# multi-line bash -c body.
sudo -u ubuntu -H bash -c '
  cd /home/ubuntu
  aws s3 cp s3://${BUCKET}/naip-bundle/bootstrap_naip.sh ./bootstrap_naip.sh
  BUCKET=${BUCKET} SHARD=${SHARD} LIMIT=${LIMIT} \
    setsid nohup bash bootstrap_naip.sh > /home/ubuntu/naip.out 2>&1 < /dev/null &
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=phase3-naip-${SHARD//\//-}}]" \
    --query 'Instances[0].InstanceId' --output text 2>&1
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  for itype in "${TYPES[@]}"; do
    for subnet in "${SUBNETS[@]}"; do
      echo "[launch] attempt $attempt: $itype / $subnet (spot, shard=$SHARD, limit=${LIMIT:-none})"
      out="$(try_launch "$itype" "$subnet")"
      if [[ "$out" == i-* ]]; then
        echo "[launch] SUCCESS: $itype $out"
        echo "$out" > /tmp/phase3_naip_instance_id.txt
        rm -f "$UD"
        exit 0
      fi
      echo "[launch]   declined: $(echo "$out" | tr -d '\n' | cut -c1-220)"
    done
  done
  echo "[launch] no spot capacity for any type/AZ — waiting 300s, then retry (NEVER on-demand)"
  sleep 300
done
