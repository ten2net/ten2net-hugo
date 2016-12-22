+++
date = "2016-12-22T11:15:53+08:00"
title = "我的docker笔记"
keywords = ["docker"]
tags = ["docker"]
topics = ["docker"]
description = "不介绍docker的安装，只记录常用的命令和一些小技巧"
author = "wangf"
draft = false
type = "post"

+++

# 1 Docker命令基础练习
```sh
docker info
docker images
docker ps
docker version
docker run hello-world
docker pull busybox
docker exec -it busybox /bin/bash

docker run -d -p 80:80 --name webserver nginx
docker run -it alpine env
```
# 2 Dockerfile使用范例

```
FROM tomcat
ADD helloworld.war /usr/local/tomcat/webapps/
EXPOSE 8080
CMD ["catalina.sh", "run"]


docker build -t mytomcat .
docker run -d -p 9280:8080 mytomcat2
```

# 3 使用 Docker， 7 个命令部署一个 Mesos 集群
  参考：https://segmentfault.com/a/1190000002531072

```

第一步：或者 Docker 服务器的 IP 并导出到环境变量。我们将在随后的 Docker 命令中不断地使用这个 IP。
set HOST_IP=10.11.31.7
第二步：启动 ZooKeeper 容器
docker run -d -p 2181:2181 -p 2888:2888 -p 3888:3888 garland/zookeeper
第三步：启动 Mesos Master
docker run --net="host" -p 5050:5050 -e "MESOS_HOSTNAME=${HOST_IP}" -e "MESOS_IP=${HOST_IP}" -e "MESOS_ZK=zk://${HOST_IP}:2181/mesos" -e "MESOS_PORT=5050" -e "MESOS_LOG_DIR=/var/log/mesos" -e "MESOS_QUORUM=1" -e "MESOS_REGISTRY=in_memory" -e "MESOS_WORK_DIR=/var/lib/mesos" -d garland/mesosphere-docker-mesos-master
第四步：启动 Marathon
docker run -d -p 8180:8180 garland/mesosphere-docker-marathon --master zk://${HOST_IP}:2181/mesos --zk zk://${HOST_IP}:2181/marathon
第五步：在一个容器中启动 Mesos Slave
docker run -d --name mesos_slave_1 --entrypoint="mesos-slave" -e "MESOS_MASTER=zk://${HOST_IP}:2181/mesos" -e "MESOS_LOG_DIR=/var/log/mesos" -e "MESOS_LOGGING_LEVEL=INFO" garland/mesosphere-docker-mesos-master:latest
第六步：进入 Mesos 的 webpage
http://${HOST_IP}:5050
第七步：进入 Marathon 的 webpage 启动一个任务
http://${HOST_IP}:8080
```

# 4 使用Docker 加速器

```sh
echo "DOCKER_OPTS=\"\$DOCKER_OPTS --registry-mirror=https://z5sa40yd.mirror.aliyuncs.com\"" | sudo tee -a /etc/default/docker sudo service docker restart
```
- 阿里云-我的专属加速器地址：https://z5sa40yd.mirror.aliyuncs.com

- 这个命令的用法忘记了
```
docker-machine create --virtualbox-no-vtx-check --engine-registry-mirror=https://z5sa40yd.mirror.aliyuncs.com -d virtualbox default
```

# 5 ui-for-docker，可视化管理Docker的工具

- Quickstart ：https://github.com/kevana/ui-for-docker

```sh
 docker run -d -p 9000:9000 --privileged -v /var/run/docker.sock:/var/run/docker.sock uifd/ui-for-docker
```

- Open your browser to http://<your Host IP>:9000
==============================================================================================



# 6 Docker Run 命令的常用选项说明

- 你的Container会在你结束命令之后自动退出，使用以下的命令选项可以将容器保持在激活状态：

> -i 即使在没有附着的情况下依然保持STDIN处于开启
> -t 分配一个伪TTY控制台

- 所以run命令就变成了：
```
docker run -it -d shykes/pybuilder bin/bash
```
# 6 Docker Exec 命令可以执行正在运行的Docker容器中的Shell命令

- 如果希望能够附着到一个已经存在的容器中，则利用exec命令：

```
docker exec -it CONTAINER_ID bash
```
# 7 常见的Docker命令行命令进行详细介绍

## 7.1 与容器（Container）相关的命令

- docker create 会创建一个容器但是不会立刻启动
- docker run 会创建并且启动某个容器
- 如果只是希望有一个暂时性的容器，可以使用 docker run --rm 将会在容器运行完毕之后删除该容器。

- 如果希望在打开某个容器之后能够与其进行交互, docker run -t -i  会创建一个TTY控制台。

- docker stop 会关闭某个容器
- docker start 会启动某个容器
- docker restart 会重新启动某个容器
- docker rm 会删除某个容器
- 如果希望能够移除所有与该容器相关的Volume，可以使用-v参数： docker rm -v.

- docker kill 会发送SIGKILL信号量到某个容器
- docker attach 会附着到某个正在运行的容器
- docker wait 会阻塞直到某个容器关闭

## 7.2 与镜像（Image）相关的命令

- docker images 会展示所有的镜像
- docker import 会从原始码中创建镜像
- docker build 会从某个Dockfile中创建镜像
- docker commit 会从某个Container中创建镜像
- docker rmi 会移除某个镜像
- docker load 以STDIN的方式从某个tar包中加载镜像
- docker save 以STDOUT的方式将镜像存入到某个tar包中

## 7.3 查看Docker容器状态信息的命令

- docker ps 会列举出所有正在运行的容器
- docker ps -a 会展示出所有正在运行的和已经停止的容器
- docker logs 从某个容器中获取log日志
- docker inspect 检测关于某个容器的详细信息
- docker events 从某个容器中获取所有的事件
- docker port 获取某个容器的全部的开放端口
- docker top 展示某个容器中运行的全部的进程
- docker stats 展示某个容器中的资源的使用情况的统计信息
- docker diff 展示容器中文件的变化情况

## 7.4 查看Docker镜像（Image）状态信息的命令

- docker history 展示镜像的全部历史信息
- docker tag 为某个容器设置标签
- Import&Export
- docker cp 在容器与本地文件系统之间进行文件复制
- docker export 将某个容器中的文件系统的内容输出到某个tar文件中

# 8 实验Machine Learning 过程中练习的命令

- docker run  ermaker/keras
- docker run -d -p 8888:8888 -e KERAS_BACKEND=tensorflow ermaker/keras-jupyter


- docker run -d -p 8888:8888 --name keraslearning  --restart=always  -v /notebook:/notebook ermaker/keras-jupyter
- docker run -d -p 8888:8888 --name keraslearning   --restart=always -v E:/python-dev-home:/notebook ermaker/keras-jupyter

- docker run -d -p 8888:8888 --name keraslearning --restart=always -v E:/python-dev-home:/notebook ermaker/keras-jupyter

- nvidia-docker run  -d -p 5001:5000 -v /dataOne:/opt --name digits --restart=always  kaixhin/cuda-digits:8.0



# 9 其它常用命令
```sh
    # 像Docker官方的hello world例子一样，拉取一个叫busybox的镜像
    docker pull busybox
    
    #进入容器bash
    docker exec -i keraslearning bash

    # 查看本地已经有哪些镜像
    # 我们可以看到busybox
    docker images

    # 现在让我们来修改下busybox镜像的容器
    # 这次，我们创建一个文件夹
    docker run busybox mkdir /home/test


    #从容器keraslearning中复制/notebook目录到当前目录
    docker cp keraslearning:/notebook .

    #从当前目录复制test子目录到容器keraslearning中/notebook目录下
    docker cp test keraslearning:/notebook

    # 让我们再看看我们有哪些镜像了。
    # 注意每条命令执行后容器都会停止
    # 可以看到有一个busybox容器
    docker ps -a

    # 现在，可以提交修改了。
    # 提交后会看到一个新的镜像busybox-1
    #  <CONTAINER ID> 是刚刚修改容器后得到的ID
    docker commit <CONTAINER ID> busybox-1

    # 再看看我们有哪些镜像。
    # 我们现在同时有busybox和busybox-1镜像了。
    docker images

    # 我们执行以下命令，看看这两个镜像有什么不同
    docker run busybox [ -d /home/test ] && echo 'Directory found' || echo 'Directory not found'
    docker run busybox-1 [ -d /home/test ] && echo 'Directory found' || echo 'Directory not found'


    # 查看所有的容器
    docker ps -a

    # 删除它们
    docker rm <CONTAINER ID>

    # 查看所有的镜像
    docker images

    # 删除它们
    docker rmi busybox-1
    docker rmi busybox

    注：可以使用 docker rm $(docker ps -q -a) 一次性删除所有的容器，docker rmi $(docker images -q) 一次性删除所有的镜像。

    #导出容器
    docker export <CONTAINER ID> -o containers/export123.tar

    #导出镜像
    docker save -o gds-keras-jupyter.tar gds/keraslearning

    #快照容器的当前状态为一个镜像
    docker commit 0d8facbc75e2 gds/keraslearning


    现在我们创建了两个Tar文件，让我们来看看它们是什么。首先做一下小清理——把所有的容器和镜像都删除：

    # 查看所有的容器
    sudo docker ps -a

    # 删除它们
    sudo docker rm <CONTAINER ID>

    # 查看所有的镜像
    sudo docker images

    # 删除它们
    sudo docker rmi busybox-1
    sudo docker rmi busybox
    注：可以使用 docker rm $(docker ps -q -a) 一次性删除所有的容器，docker rmi $(docker images -q) 一次性删除所有的镜像。

    现在开始导入刚刚导出的容器：

    # 导入export.tar文件
    cat /home/export.tar | sudo docker import - busybox-1-export:latest

    # 查看镜像
    sudo docker images

    # 检查是否导入成功，就是启动一个新容器，检查里面是否存在/home/test目录（是存在的）
    sudo docker run busybox-1-export [ -d /home/test ] && echo 'Directory found' || echo 'Directory not found'
    使用类似的步骤导入镜像：

    # 导入save.tar文件
    docker load < /home/save.tar
    docker load -i /home/save.tar

    # 查看镜像
    sudo docker images

    # 检查是否导入成功，就是启动一个新容器，检查里面是否存在/home/test目录（是存在的）
    sudo docker run busybox-1 [ -d /home/test ] && echo 'Directory found' || echo 'Directory not found'
    那，它们之间到底存在什么不同呢？我们发现导出后的版本会比原来的版本稍微小一些。那是因为导出后，会丢失历史和元数据。执行下面的命令就知道了：

    # 显示镜像的所有层(layer)
    sudo docker images --tree
     执行命令，显示下面的内容。正你看到的，导出后再导入(exported-imported)的镜像会丢失所有的历史，而保存后再加载（saveed-loaded）的镜像没有丢失历史和层(layer)。这意味着使用导出后再导入的方式，你将无法回滚到之前的层(layer)，同时，使用保存后再加载的方式持久化整个镜像，就可以做到层回滚（可以执行docker tag <LAYER ID> <IMAGE NAME>来回滚之前的层）。

    sudo docker images --tree
    ├─f502877df6a1 Virtual Size: 2.489 MB Tags: busybox-1-export:latest
    └─511136ea3c5a Virtual Size: 0 B
      └─bf747efa0e2f Virtual Size: 0 B
        └─48e5f45168b9 Virtual Size: 2.489 MB
          └─769b9341d937 Virtual Size: 2.489 MB
            └─227516d93162 Virtual Size: 2.489 MB Tags: busybox-1:latest

```