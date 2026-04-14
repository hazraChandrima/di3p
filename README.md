## DI3P

### Clone & Build

```bash
git clone https://github.com/hazraChandrima/di3p
cd di3p/

mkdir build && cd build

cmake ..
make
```

### Run

Inside the `build/` directory:

```bash
./analyzer <source> <dest_dirname> [K=4] [--nodct]
```

| arg | desc |
|---|---|
| `source` | Path to the input image |
| `dest_dirname` | Name of the output folder — program writes to `../images/<dest_dirname>/output/` |
| `K` | Number of segments for K-Means (default: `4`) |
| `--nodct` | Skip DCT blockiness |


## Results

### 1. Flower


<div align="center">
  <div align="center">Input Image<br/><i>kuch yaad ayi?</i></div>
  <img src="./images/test1/flower.jpeg" alt="Input Image" width="400"/>
</div>

<br/>


| K | Segmentation | Output Image | Evaluation Metrics |
|:-:|:---:|:---:|:---:|
| **4** | <img src="./images/test1/output/k4/k4.jpg" width="400"/> | <img src="./images/test1/output/k4/enhanced-k4.jpg" width="400"/> | <img src="./images/test1/output/k4/eval.png" width="400"/> |
| **6** | <img src="./images/test1/output/k6/k6.jpg" width="400"/> | <img src="./images/test1/output/k6/enhanced-k6.jpg" width="400"/> | <img src="./images/test1/output/k6/eval.png" width="400"/> |
| **10** | <img src="./images/test1/output/k10/k10.jpg" width="400"/> | <img src="./images/test1/output/k10/enhanced-k10.jpg" width="400"/> | <img src="./images/test1/output/k10/eval.png" width="400"/> |
| **20** | <img src="./images/test1/output/k20/k20.jpg" width="400"/> | <img src="./images/test1/output/k20/enhanced-k20.jpg" width="400"/> | <img src="./images/test1/output/k20/eval.png" width="400"/> |
| **40** | <img src="./images/test1/output/k40/k40.jpg" width="400"/> | <img src="./images/test1/output/k40/enhanced-k40.jpg" width="400"/> | <img src="./images/test1/output/k40/eval.png" width="400"/> |


### 2. Hills

<div align="center">
  <div align="center">Input Image</div>
  <img src="./images/test2/scene.jpeg" alt="Input Image" width="400"/>
</div>

<br/>

| K | Segmentation | Output Image | Evaluation Metrics |
|:-:|:---:|:---:|:---:|
| **2** | <img src="./images/test2/output/k2/k2.jpg" width="400"/> | <img src="./images/test2/output/k2/enhanced-k2.jpg" width="400"/> | <img src="./images/test2/output/k2/eval.png" width="400"/> |
| **4** | <img src="./images/test2/output/k4/k4.jpg" width="400"/> | <img src="./images/test2/output/k4/enhanced-k4.jpg" width="400"/> | <img src="./images/test2/output/k4/eval.png" width="400"/> |
| **6** | <img src="./images/test2/output/k6/k6.jpg" width="400"/> | <img src="./images/test2/output/k6/enhanced-k6.jpg" width="400"/> | <img src="./images/test2/output/k6/eval.png" width="400"/> |
| **10** | <img src="./images/test2/output/k10/k10.jpg" width="400"/> | <img src="./images/test2/output/k10/enhanced-k10.jpg" width="400"/> | <img src="./images/test2/output/k10/eval.png" width="400"/> |
| **20** | <img src="./images/test2/output/k20/k20.jpg" width="400"/> | <img src="./images/test2/output/k20/enhanced-k20.jpg" width="400"/> | <img src="./images/test2/output/k20/eval.png" width="400"/> |
| **40** | <img src="./images/test2/output/k40/k40.jpg" width="400"/> | <img src="./images/test2/output/k40/enhanced-k40.jpg" width="400"/> | <img src="./images/test2/output/k40/eval.png" width="400"/> |
