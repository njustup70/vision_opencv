## 容器构建
- 支持 --cpu 或者 --ros 与无参数，对应只用cpu(在维护) ，ros2耦合与 gpu(较大)
- 使用历程
```bash
.devcontainer$ ./build.sh --cpu 
```
## 模块介绍
|模块 |说明 |
|---|---|
|[cv_node](./cv_node/)|具体逻辑实现,类似ros2节点(核心代码)|
|[cv_lib](./cv_lib/)|每个功能的实现|
|[PoseSolver](./PoseSolver/)|pnp与aruco码|
|[YOLOv11](./YOLOv11/)|YOLO代码都在里面|
## 开发容器
- 进入容器后需要选择解释器(在vscode右下角)
- 打开终端输入
```bash
which python
```
- 选择同一个路径的python就行

## 注意事项
### 现在的数据源从ros2话题获得,并且需要在ros2那边开一个转发节点
