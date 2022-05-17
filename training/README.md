# training

## AWS EC2

### Instance Configuration

Here are the settings used to create our EC2 instance:

- AMI: Deep Learning AMI (Ubuntu 18.04) Version 50.0
- Instance Type: p2.xlarge
  - 4 CPU, 61 GiB memory
- Storage: 1x 100 GiB gp2

### `ssh` Set Up

```bash
chmod 400 filename.pem
ssh-add filename.pem
ssh ubuntu@ec2-11-111-11-11.your-region.compute.amazonaws.com
# Type in `yes` to prompt
```

#### Adding More Key Pairs

Here is how to add more key pairs for teammates:

1. Follow the AWS tutorial [Create key pairs][1] to create a new key pair
   - Key pair type: ED25519
   - Private key format: .pem
1. Run these commands:

```bash
chmod 400 filename.pem
ssh-keygen -f filename.pem -y > filename.pub
ssh ubuntu@ec2-11-111-11-11.your-region.compute.amazonaws.com
nano ~/.ssh/authorized_keys
```

3. Append the contents of `filename.pub` as well as `filename`
1. Share the `filename.pem` with your peer

### Instance Set Up

```bash
sudo apt update -y
sudo apt install libsqlite3-dev  # SEE: https://stackoverflow.com/a/38842600/11163122
sudo apt autoremove -y
curl https://pyenv.run | bash
# Follow instructions on what to append to ~/.bashrc
exec $SHELL
pyenv install 3.10:latest
pyenv local 3.10.4  # Or whatever version was installed
```

Then I followed the instructions in the repo root's `README.md`.

[1]: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/create-key-pairs.html#having-ec2-create-your-key-pair
