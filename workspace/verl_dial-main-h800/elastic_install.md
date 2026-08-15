参考 [Elastic 官方安装文档](https://www.elastic.co/docs/deploy-manage/deploy/self-managed/install-elasticsearch-from-archive-on-linux-macos)

**1. 下载并解压**
以 `root` 执行：

```bash
apt-get update
apt-get install -y curl ca-certificates

cd /tmp
curl -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-9.4.2-linux-x86_64.tar.gz
curl -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-9.4.2-linux-x86_64.tar.gz.sha512

sha512sum -c elasticsearch-9.4.2-linux-x86_64.tar.gz.sha512

tar -xzf elasticsearch-9.4.2-linux-x86_64.tar.gz -C /opt
ln -sfn /opt/elasticsearch-9.4.2 /opt/elasticsearch
```


**2. 创建专用用户和目录**
Elasticsearch 不能使用 `root` 运行：

```bash
groupadd --system elasticsearch
useradd --system \
  --gid elasticsearch \
  --home-dir /opt/elasticsearch \
  --shell /usr/sbin/nologin \
  elasticsearch

mkdir -p /var/lib/elasticsearch
mkdir -p /var/log/elasticsearch
mkdir -p /run/elasticsearch

chown -R elasticsearch:elasticsearch /opt/elasticsearch-9.4.2
chown -R elasticsearch:elasticsearch /var/lib/elasticsearch
chown -R elasticsearch:elasticsearch /var/log/elasticsearch
chown -R elasticsearch:elasticsearch /run/elasticsearch
```

如果提示用户或用户组已存在，可以忽略对应错误。

**3. 配置 Elasticsearch**
编辑（或者用vscode编辑器打开这个文件来编辑）：

```bash
vi /opt/elasticsearch/config/elasticsearch.yml
```

写入：

```yaml
cluster.name: clarq-search
node.name: clarq-es01

discovery.type: single-node
network.host: 127.0.0.1
http.port: 9200

path.data: /var/lib/elasticsearch
path.logs: /var/log/elasticsearch

xpack.security.enabled: true
xpack.security.autoconfiguration.enabled: false
xpack.security.http.ssl.enabled: false
```

配置 `2 GB` 堆内存：

```bash
mkdir -p /opt/elasticsearch/config/jvm.options.d

printf '%s\n' '-Xms2g' '-Xmx2g' \
  > /opt/elasticsearch/config/jvm.options.d/heap.options

chown -R elasticsearch:elasticsearch /opt/elasticsearch/config
```

**4. 设置管理员密码**
不要继续使用原配置中的弱密码。生成一个新密码：

```bash
export ELASTIC_PASSWORD='替换成一个新的强密码'（建议写成 123456）

printf '%s' "$ELASTIC_PASSWORD" |
  runuser -u elasticsearch -- \
  /opt/elasticsearch/bin/elasticsearch-keystore add -x bootstrap.password
```

**5. 启动**
容器里没有 `systemd`，使用 Elasticsearch 自带的后台模式：

```bash
ulimit -n 65536

runuser -u elasticsearch -- \
  /opt/elasticsearch/bin/elasticsearch \
  -d -p /run/elasticsearch/elasticsearch.pid
```

官方支持用 `-d -p` 启动压缩包版本。[启动与停止文档](https://www.elastic.co/docs/deploy-manage/maintenance/start-stop-services/start-stop-elasticsearch)

查看日志：

```bash
tail -f /var/log/elasticsearch/clarq-search.log
```

验证：

```bash
curl -u "elastic:${ELASTIC_PASSWORD}" \
  http://127.0.0.1:9200/_cluster/health?pretty
```

停止服务：

```bash
kill "$(cat /run/elasticsearch/elasticsearch.pid)"
```