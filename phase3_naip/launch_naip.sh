#!/usr/bin/env bash
# Launch the Phase 3 NAIP worker on a SPOT GPU instance.
#
# SPOT ONLY — never on-demand (per [[never_on_demand]]). g6e.2xlarge primary
# (L40S 48 GB VRAM, 8 vCPU). 2026-05-25 telemetry confirms steady-state GPU
# 100% / CPU 40%: SAM forward IS the bottleneck. Stage-summed postproc looks
# big (62%) because it's masured across all clusters, but the ProcessPool
# parallelizes it under 8 procs to a wall floor below SAM's serialized critical
# path. Bigger vCPU counts don't move the wall.
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
TYPES=(g6e.2xlarge)  # GPU-bound on L40S; CPU 40% during steady-state (not saturated). Bigger vCPU box gains nothing.

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
