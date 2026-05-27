#!/usr/bin/env bash
# Launch a CPU spot for elbow_scan.py — L7 2008 SR 6-band signature.
# Same envelope as launch.sh but runs bootstrap_elbow.sh.
set -uo pipefail
cd "$(dirname "$0")/../../.."

[ -f sites_us/.env.agent-profile ] && set -a && . sites_us/.env.agent-profile && set +a
[ -f sites_us/.env ] && set -a && . sites_us/.env && set +a

REGION="${AWS_REGION:-us-west-2}"
BUCKET="${AWS_S3_RESULTS_BUCKET:?AWS_S3_RESULTS_BUCKET not set}"
KEY="industrials-scanner-key"
SG="sg-0e344ef6a61de3d56"
PROFILE="industrials-scanner-profile"
SUBNETS=(subnet-0463b6600119e939d subnet-0c177d1da88790caa subnet-0e7a01003e03380f8 subnet-096bbbe80d8ea7409)
TYPES=(c6a.4xlarge c6i.4xlarge m6a.4xlarge c6a.8xlarge c6i.8xlarge)

AMI=$(aws ssm get-parameters --region "$REGION" \
  --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
  --query 'Parameters[0].Value' --output text 2>/dev/null)
if [[ "$AMI" != ami-* ]]; then
  echo "[launch-elbow] ERROR: could not resolve Ubuntu 22.04 AMI via SSM"; exit 1
fi
echo "[launch-elbow] using AMI: $AMI"

UD=$(mktemp)
cat > "$UD" <<EOF
#!/bin/bash
exec > /var/log/user-data.log 2>&1
set -x
sudo apt-get update -qq && sudo apt-get install -y -qq awscli python3 python3-pip python3-venv build-essential libgdal-dev
sudo -u ubuntu -H bash -c '
  cd /home/ubuntu
  aws s3 cp s3://${BUCKET}/v3-pt2-bundle/bootstrap_elbow.sh ./bootstrap_elbow.sh
  chmod +x bootstrap_elbow.sh
  BUCKET=${BUCKET} STAGE2B_RUN_TAG="${STAGE2B_RUN_TAG:-}" STAGE2B_MIN_PROB="${STAGE2B_MIN_PROB:-0.30}" STAGE2B_MAX_PROB="${STAGE2B_MAX_PROB:-1.01}" setsid nohup bash bootstrap_elbow.sh > /home/ubuntu/v3pt2.out 2>&1 < /dev/null &
'
EOF

try_launch() {
  aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$1" \
    --key-name "$KEY" --security-group-ids "$SG" --subnet-id "$2" \
    --iam-instance-profile "Name=$PROFILE" \
    --instance-initiated-shutdown-behavior terminate \
    --instance-market-options 'MarketType=spot' \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=50,VolumeType=gp3}' \
    --user-data "file://$UD" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=stage2b-elbow-scan}]" \
    --query 'Instances[0].InstanceId' --output text 2>&1
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  for itype in "${TYPES[@]}"; do
    for subnet in "${SUBNETS[@]}"; do
      echo "[launch-elbow] attempt $attempt: $itype / $subnet (spot)"
      out="$(try_launch "$itype" "$subnet")"
      if [[ "$out" == i-* ]]; then
        echo "[launch-elbow] SUCCESS: $itype $out"
        echo "$out" > /tmp/elbow_instance_id.txt
        echo
        echo "Stream log:  aws s3 cp s3://${BUCKET}/v3-pt2-artifacts/logs/${out}.log - | tail -f"
        echo "Heartbeat:   aws s3 cp s3://${BUCKET}/v3-pt2-artifacts/heartbeat/${out}.json -"
        rm -f "$UD"
        exit 0
      fi
      echo "[launch-elbow]   declined: $(echo "$out" | tr -d '\n' | cut -c1-200)"
    done
  done
  echo "[launch-elbow] no spot capacity any type/AZ — waiting 300s (NEVER on-demand)"
  sleep 300
done
