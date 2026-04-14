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
./analyzer ../images/fields.jpg [K=4]
```

---

## Sample Results

### Input Image

<p align="center">
  <img src="./images/test1/flower.jpeg" alt="Input Image" width="400"/>
</p>

---

## Results

| K | Segmentation | Enhanced Output | Evaluation Metrics |
|:-:|:---:|:---:|:---:|
| **4** | <img src="./images/test1/output/k4/k4.jpg" width="200"/> | <img src="./images/test1/output/k4/enhanced-k4.jpg" width="200"/> | <img src="./images/test1/output/k4/eval.png" width="200"/> |
| **6** | <img src="./images/test1/output/k6/k6.jpg" width="200"/> | <img src="./images/test1/output/k6/enhanced-k6.jpg" width="200"/> | <img src="./images/test1/output/k6/eval.png" width="200"/> |
| **10** | <img src="./images/test1/output/k10/k10.jpg" width="200"/> | <img src="./images/test1/output/k10/enhanced-k10.jpg" width="200"/> | <img src="./images/test1/output/k10/eval.png" width="200"/> |
| **14** | <img src="./images/test1/output/k14/k14.jpg" width="200"/> | <img src="./images/test1/output/k14/enhanced-k14.jpg" width="200"/> | <img src="./images/test1/output/k14/eval.png" width="200"/> |
| **20** | <img src="./images/test1/output/k20/k20.jpg" width="200"/> | <img src="./images/test1/output/k20/enhanced-k20.jpg" width="200"/> | <img src="./images/test1/output/k20/eval.png" width="200"/> |
