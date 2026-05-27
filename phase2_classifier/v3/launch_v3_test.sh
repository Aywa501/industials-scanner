#!/usr/bin/env bash
# Launch v3 test-set scoring on a small spot GPU.
# g4dn.2xlarge (8 vCPU, 32 GB, T4) primary; g4dn.xlarge (4 vCPU, 16 GB)
# fallback if 2xlarge spot exhausted (model + 2K crops fit, tighter).
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
TYPES=(g4dn.2xlarge g4dn.xlarge)

UD="$(mktemp)"
cat > "$UD" <<EOF
#!/bin/bash
exec > /var/log/user-data.log 2>&1
set -x
sudo -u ubuntu -H bash -c '
  cd /home/ubuntu
  aws s3 cp s3://${BUCKET}/v3-test-bundle/bootstrap_v3_test.sh ./bootstrap_v3_test.sh
  aws s3 cp s3://${BUCKET}/v3-test-bundle/.env ./.env
  BUCKET=${BUCKET} \
    setsid nohup bash bootstrap_v3_test.sh > /home/ubuntu/v3_test.out 2>&1 < /dev/null &
'
EOF

try_launch() {
  aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$1" \
    --key-name "$KEY" --security-group-ids "$SG" --subnet-id "$2" \
    --iam-instance-profile "Name=$PROFILE" \
    --instance-initiated-shutdown-behavior terminate \
    --instance-market-options 'MarketType=spot' \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
    --user-data "file://$UD" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=v3-test}]" \
    --query 'Instances[0].InstanceId' --output text 2>&1
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  for itype in "${TYPES[@]}"; do
    for subnet in "${SUBNETS[@]}"; do
      echo "[launch-test] attempt $attempt: $itype / $subnet (spot)"
      out="$(try_launch "$itype" "$subnet")"
      if [[ "$out" == i-* ]]; then
        echo "[launch-test] SUCCESS: $itype $out"
        echo "$out" > /tmp/v3_test_instance_id.txt
        rm -f "$UD"
        exit 0
      fi
      echo "[launch-test]   declined: $(echo "$out" | tr -d '\n' | cut -c1-220)"
    done
  done
  echo "[launch-test] no spot any type/AZ — waiting 300s (NEVER on-demand)"
  sleep 300
done
