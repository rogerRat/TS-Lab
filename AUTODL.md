# Data storage

Read [AGENTS.md](AGENTS.md) first: it defines the repository-wide storage policy.
Keep the core Git repository (code, configuration, scripts, tests and documentation)
under `/root`, preferably `/root/TS3`. Put large datasets, weights, checkpoints,
outputs and caches on `/root/autodl-tmp`, using explicit paths or project symlinks.
Here TMP means the data disk, not `/tmp`; experiment results there must not be
treated as disposable files. Existing deployments must not be moved without approval.
The `autodl-pub` folder is read-only. Do not use `autodl-fs` by default.
If disk capacity is insufficient, ask the user to expand it or provide another data disk.

# Network proxy
If you need to fetch or download anything from GitHub or Hugging Face, please use the following command before downloading:
```
source /etc/network_turbo
```
Since this can give a better network performance When accessing these websites, this can reduce the bandwidth for visiting domestic(China mailand) websites.
When no longer needed, please use:
```
unset http_proxy && unset https_proxy
```
To unset the network proxy.

For huggingface, you may need to download large-scale data, you can try:
```
export HF_ENDPOINT=https://hf-mirror.com
```

# Command running and auto-shutdown
There can be scenarios when the user may need to run a specific script and shut down the docker immediately when it exits to save the GPU hours.
Under that circumstance, you can use:
```
/usr/bin/shutdown
```
To auto shutdown the docker.

example:
```
# Original command:
python train.py
# ->
python train.py; /usr/bin/shutdown      # ; Means whenever the command success or not, the docker shuts down.
python train.py && /usr/bin/shutdown    # && Means only when the command success, the docker shuts down.
```
