#!/usr/bin/env bash
# Launch the v3 detector training on a SPOT GPU instance.
#
# SPOT ONLY (per [[never_on_demand]]). v3 is I/O-bound on NAIP-COG S3 reads, not
# GPU-bound — we want network bandwidth within the 32-vCPU spot quota.
# Order: g4dn.8xlarge (T4, 32 vCPU, 50 Gbps up-to, 128 GB) → g6.8xlarge (L4, 32 vCPU,
# 25 Gbps). NO other sizes/families (user rule).
#
# Env knobs forwarded to bootstrap:
#   V3_PREFLIGHT_ONLY  — if set to N, run preflight on N buildings and exit
#                        (useful for capturing in-region fetch rate before
#                        committing to the full embed pass)
set -uo pipefail
cd "$(dirname "$0")/../../.."

# AWS management creds (separate from runtime .env IAM user that talks to naip-analytic)
[ -f sites_us/.env.agent-profile ] && set -a && . sites_us/.env.agent-profile && set +a
[ -f sites_us/.env ] && set -a && . sites_us/.env && set +a

REGION="${AWS_REGION:-us-west-2}"
BUCKET="${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set}"
AMI="ami-0b1c3e116f347cc8a"          # DLAMI OSS Nvidia PyTorch 2.6, Ubuntu 22.04
KEY="industrials-scanner-key"
SG="sg-0e344ef6a61de3d56"
PROFILE="industrials-scanner-profile"
SUBNETS=(subnet-096bbbe80d8ea7409 subnet-0463b6600119e939d subnet-0c177d1da88790caa subnet-0e7a01003e03380f8)
TYPES=(g4dn.8xlarge g6.8xlarge)

V3_PREFLIGHT_ONLY="${V3_PREFLIGHT_ONLY:-}"
TAG_SUFFIX="full"
[ -n "$V3_PREFLIGHT_ONLY" ] && TAG_SUFFIX="preflight-${V3_PREFLIGHT_ONLY}"

UD="$(mktemp)"
cat > "$UD" <<EOF
#!/bin/bash
exec > /var/log/user-data.log 2>&1
set -x
sudo -u ubuntu -H bash -c '
  cd /home/ubuntu
  aws s3 cp s3://${BUCKET}/v3-bundle/bootstrap_v3.sh ./bootstrap_v3.sh
  aws s3 cp s3://${BUCKET}/v3-bundle/.env ./.env
  BUCKET=${BUCKET} V3_PREFLIGHT_ONLY=${V3_PREFLIGHT_ONLY} \
    setsid nohup bash bootstrap_v3.sh > /home/ubuntu/v3_train.out 2>&1 < /dev/null &
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=v3-${TAG_SUFFIX}}]" \
    --query 'Instances[0].InstanceId' --output text 2>&1
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  for itype in "${TYPES[@]}"; do
    for subnet in "${SUBNETS[@]}"; do
      echo "[launch] attempt $attempt: $itype / $subnet (spot, mode=${TAG_SUFFIX})"
      out="$(try_launch "$itype" "$subnet")"
      if [[ "$out" == i-* ]]; then
        echo "[launch] SUCCESS: $itype $out"
        echo "$out" > /tmp/v3_instance_id.txt
        echo "$itype" > /tmp/v3_instance_type.txt
        rm -f "$UD"
        exit 0
      fi
      echo "[launch]   declined: $(echo "$out" | tr -d '\n' | cut -c1-220)"
    done
  done
  echo "[launch] no spot capacity any type/AZ — waiting 300s (NEVER on-demand)"
  sleep 300
done
