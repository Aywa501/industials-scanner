#!/usr/bin/env bash
# Launch the Phase 3 v2 CONUS re-scan on a SPOT GPU instance.
#
# SPOT ONLY — never on-demand. g6.8xlarge ONLY for this run (g6.4xlarge
# fallback removed by request). Tries every default-VPC AZ in turn; if every
# AZ capacity-fails, waits 5 min and retries the whole sweep — it never falls
# back to on-demand or to another instance type.
#
# The instance auto-runs bootstrap_v2_scan.sh via user-data and self-terminates
# on completion or failure (bootstrap EXIT trap + instance-role TerminateInstances).
set -uo pipefail
cd "$(dirname "$0")/../../.."
[ -f sites_us/.env ] && set -a && . ./sites_us/.env && set +a

REGION="${AWS_REGION:-us-west-2}"
BUCKET="${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set}"
AMI="ami-0b1c3e116f347cc8a"          # DLAMI OSS Nvidia PyTorch 2.6, Ubuntu 22.04
KEY="industrials-scanner-key"
SG="sg-0e344ef6a61de3d56"
PROFILE="industrials-scanner-profile"
SUBNETS=(subnet-096bbbe80d8ea7409 subnet-0463b6600119e939d subnet-0c177d1da88790caa subnet-0e7a01003e03380f8)
TYPES=(g6.8xlarge)

UD="$(mktemp)"
cat > "$UD" <<EOF
#!/bin/bash
exec > /var/log/user-data.log 2>&1
set -x
# Run the bootstrap as ubuntu. Use 'sudo -u ubuntu -H', NOT 'sudo -iu ubuntu':
# the -i (login) flag makes sudo re-quote the command through the login shell's
# -c, which eats escaped newlines as line-continuations — collapsing the multi-
# line bash -c body into one line so 'cd' swallows every following word
# ('cd: too many arguments'). -u -H preserves newlines and still sets HOME.
sudo -u ubuntu -H bash -c '
  cd /home/ubuntu
  aws s3 cp s3://${BUCKET}/scan-v2-bundle/bootstrap_v2_scan.sh ./bootstrap_v2_scan.sh
  BUCKET=${BUCKET} setsid nohup bash bootstrap_v2_scan.sh > /home/ubuntu/scan_v2.out 2>&1 < /dev/null &
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
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=phase3-scan-v2}]' \
    --query 'Instances[0].InstanceId' --output text 2>&1
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  for itype in "${TYPES[@]}"; do
    for subnet in "${SUBNETS[@]}"; do
      echo "[launch] attempt $attempt: $itype / $subnet (spot)"
      out="$(try_launch "$itype" "$subnet")"
      if [[ "$out" == i-* ]]; then
        echo "[launch] SUCCESS: $itype $out"
        echo "$out" > /tmp/phase3_v2_instance_id.txt
        rm -f "$UD"
        exit 0
      fi
      echo "[launch]   declined: $(echo "$out" | tr -d '\n' | cut -c1-220)"
    done
  done
  echo "[launch] no spot capacity for any type/AZ — waiting 300s, then retry (NEVER on-demand)"
  sleep 300
done
