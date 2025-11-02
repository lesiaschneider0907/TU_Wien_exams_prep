# 📘 Lecture 3 — Classification & Neural Networks (2024W)

> **Тема:** многоклассовая классификация (**multiclass classification**), softmax-классификатор, **cross-entropy loss**, полносвязные нейросети (**feed-forward neural networks**), backpropagation, autodiff, регуляризация, и оптимизация.

---

## 0) Постановка задачи классификации (classification setup & notation)

- Дано пространство признаков \( \mathcal{X}\subseteq \mathbb{R}^d \), метки классов \( \mathcal{Y}=\{1,\dots,K\} \).
- Датасет \( \mathcal{D}=\{(x^{(i)},y^{(i)})\}_{i=1}^N \), где \( y^{(i)} \in \{1,\dots,K\} \).
- Цель: построить скоринговую функцию \( f: \mathbb{R}^d \to \mathbb{R}^K \) и правило предсказания класса  
  \[
  \hat{y}(x) = \arg\max_{k\in\{1,\dots,K\}} f_k(x).
  \]

---

## 1) Интуиция softmax-классификатора (softmax classifier intuition)

- Для многоклассового случая удобно нормировать скор \( s_k(x) \) в **распределение вероятностей** по классам:
  \[
  p_\theta(y=k\mid x)=\frac{\exp(s_k(x))}{\sum_{j=1}^K \exp(s_j(x))},\quad
  s(x)=W x + b,\; W\in\mathbb{R}^{K\times d}.
  \]
- **Softmax** превращает произвольные «сырые» счёты (**logits**) в вероятности.

---

## 2) Детали softmax и кросс-энтропии (details & cross-entropy)

### 2.1 Softmax
\[
\operatorname{softmax}(z)_k=\frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}},\qquad z\in\mathbb{R}^K.
\]

Сдвиг на константу не меняет softmax: \(\operatorname{softmax}(z)=\operatorname{softmax}(z+c\mathbf{1})\) → полезно для численной стабильности.

### 2.2 One-hot и отрицательное лог-правдоподобие
Пусть истинная метка как one-hot \(y\in\{0,1\}^K\), тогда лог-правдоподобие:
\[
\log p_\theta(y\mid x) = \sum_{k=1}^K y_k \log \operatorname{softmax}(s(x))_k.
\]

### 2.3 Потеря кросс-энтропии (cross-entropy loss)
\[
\mathcal{L}_{\text{CE}}(x,y)
= -\sum_{k=1}^K y_k \log p_\theta(y=k\mid x)
= -\log p_\theta(y^\star\mid x),
\]
где \(y^\star\) — истинный класс. На всём датасете:
\[
J(\theta) = \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}_{\text{CE}}\big(x^{(i)},y^{(i)}\big).
\]

---

## 3) Градиенты для softmax + CE (ключевая формула)

Обозначим \(z = s(x)=Wx+b\), \(p=\operatorname{softmax}(z)\).
Для одного примера и one-hot \(y\):

- По логитам:
  \[
  \frac{\partial \mathcal{L}}{\partial z_k} = p_k - y_k.
  \]
- По параметрам:
  \[
  \frac{\partial \mathcal{L}}{\partial W} = (p - y)\, x^\top,\qquad
  \frac{\partial \mathcal{L}}{\partial b} = p - y.
  \]
- По входу (для backprop сквозь слои):
  \[
  \frac{\partial \mathcal{L}}{\partial x} = W^\top (p - y).
  \]

> Именно эта компактная форма «\(p-y\)» — краеугольный камень обучения классификаторов.

**PyTorch-скелет (один батч):**
```python
import torch
logits = X @ W.T + b        # [B, K]
loss = torch.nn.functional.cross_entropy(logits, y_true)  # y_true: [B] с индексами классов
loss.backward()             # автодиф посчитает p - y и градиенты
```
---

## 4) Нейрон и нелинейность (artificial neuron & nonlinearity)

### 4.1 Логистическая регрессия как нейрон (binary)
\[
\hat{y}=\sigma(w^\top x + b),\quad \sigma(t)=\frac{1}{1+e^{-t}}.
\]

### 4.2 Полносвязный слой (fully connected / affine layer)
\[
h = f(Wx+b),\quad f\ \text{— нелинейность (activation)}.
\]

### 4.3 Зачем нужна нелинейность
Без \(f\) композиция слоёв — линейное преобразование → **не увеличивает выразительность**.
Популярные \(f\): **ReLU**, **tanh**, **GELU**, **SiLU/Swish**.

---

## 5) Полносвязная нейросеть (feed-forward neural network)

### 5.1 Архитектура с одним скрытым слоем
\[
\begin{aligned}
h &= f(W_1 x + b_1),\quad h\in\mathbb{R}^m,\\
z &= W_2 h + b_2,\quad z\in\mathbb{R}^K,\\
p &= \operatorname{softmax}(z),\quad
\mathcal{L}(x,y)=-\log p_{y^\star}.
\end{aligned}
\]

### 5.2 Backprop (цепное правило)
Пусть \(g_z=\partial \mathcal{L}/\partial z = p-y\). Тогда
\[
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W_2} &= g_z\, h^\top,\qquad
\frac{\partial \mathcal{L}}{\partial b_2} = g_z,\\[4pt]
g_h &= W_2^\top g_z,\\
\frac{\partial \mathcal{L}}{\partial W_1} &= (g_h \odot f'(W_1 x+b_1))\, x^\top,\\
\frac{\partial \mathcal{L}}{\partial b_1} &= g_h \odot f'(W_1 x+b_1).
\end{aligned}
\]

### 5.3 PyTorch: минимальный MLP-классификатор
```python
import torch, torch.nn as nn, torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_in, d_hid, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, n_classes)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits

model = MLP(d_in=300, d_hid=256, n_classes=10)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for Xb, yb in loader:
    opt.zero_grad()
    logits = model(Xb)                # [B, K]
    loss = F.cross_entropy(logits, yb)
    loss.backward()
    opt.step()
```

---

## 6) Пример: NER как классификация по окну (windowed NER)

В **Named Entity Recognition (NER)** мы классифицируем каждый токен: например, LOCATION vs NOT-LOCATION или формат IOB (B-LOC, I-LOC, O).

**Оконный подход (window classifier):**

1) Берём окно \(2C+1\) слов вокруг текущего токена \(w_t\)
2) Конкатенируем их эмбеддинги
3) Подаём в MLP + softmax

\[
x_t = [\,e(w_{t-C});\dots;e(w_t);\dots;e(w_{t+C})\,]
\in \mathbb{R}^{(2C+1)d}
\]

\[
p(y_t \mid x_t) = \operatorname{softmax}(W_2 \, f(W_1 x_t + b_1) + b_2)
\]

Простой, но мощный базовый метод (до RNN/Transformer эпохи).

---

## 7) Вычислительные графы и Backprop

Модель — это **граф вычислений**: узлы = операции (matmul, add, exp, log), рёбра = тензоры.

**Цепное правило:**

\[
\frac{dq}{dx} = 
\frac{\partial q}{\partial u}\frac{du}{dx} +
\frac{\partial q}{\partial v}\frac{dv}{dx}
\]

**Backpropagation** = применение цепного правила от выхода к входам.

### Автодифференцирование (autodiff)

- PyTorch/TF/JAX используют **reverse-mode AD**
- `loss.backward()` строит градиенты автоматически

---

## 8) Нелинейности (activation functions)

| Функция | Формула | Плюсы | Минусы |
|---|---|---|---|
| ReLU | \(f(x)=\max(0,x)\) | быстро, стабильно | dying ReLU |
| tanh | \(\frac{e^x - e^{-x}}{e^x + e^{-x}}\) | центр 0 | затухающие градиенты |
| GELU | smooth ReLU-like | современная альтернатива | дороже |
| SiLU/Swish | \(x \sigma(x)\) | часто ↑ качество | дороже |

---

## 9) Регуляризация (regularization)

- **L2 / weight decay**: \(\frac{\lambda}{2}\lVert\theta\rVert_2^2\)
- **Dropout**: маскируем часть активаций

Тренировка:  
\[
h' = \frac{m \odot h}{1-p},\quad m \sim \text{Bernoulli}(1-p)
\]

Инференс:  
\[
h' = h
\]

**PyTorch пример Dropout:**
```python
class MLPDrop(nn.Module):
    def __init__(self, d_in, d_hid, n_classes, p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.drop = nn.Dropout(p)
        self.fc2 = nn.Linear(d_hid, n_classes)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.drop(h)
        return self.fc2(h)
```
---

## 10) Векторизация (vectorization)

Избегаем Python-циклов по одному примеру → используем **batch**-вычисления.

**Медленно (плохо):**
```python
import torch
import torch.nn.functional as F

N, d, K = 1024, 300, 10
X = torch.randn(N, d)
W = torch.randn(K, d)
b = torch.randn(K)
y = torch.randint(0, K, (N,))

loss_sum = 0.0
for i in range(N):
    logits_i = X[i] @ W.T + b
    loss_i = F.cross_entropy(logits_i.unsqueeze(0), y[i].unsqueeze(0))
    loss_sum += loss_i.item()
```

**Быстро (правильно):**

```python
import torch
import torch.nn.functional as F

N, d, K = 1024, 300, 10
X = torch.randn(N, d)
W = torch.randn(K, d)
b = torch.randn(K)
y = torch.randint(0, K, (N,))

logits = X @ W.T + b           # [N, K]
loss = F.cross_entropy(logits, y)
loss.backward()
```

## 11) Инициализация параметров (parameter initialization)

Слишком маленькая дисперсия → затухающие градиенты.  
Слишком большая → взрывы градиентов / нестабильность обучения.

### Xavier / Glorot (для tanh/linear)

\[
W_{ij} \sim \mathcal{U}\!\Big(
-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},
\ \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}
\Big)
\]

### He / Kaiming (для ReLU)

```python
import torch.nn as nn

def init_kaiming_relu(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)
```

### Использование:
```python
model.apply(init_kaiming_relu)
```

## 12) Оптимизация (optimizers & LR schedules)

Основные оптимизаторы:

- **SGD + momentum**
- **Adam / AdamW** (часто лучший старт)
- **RMSProp**

Полезные фичи обучения:

- **Weight decay** (L2-регуляризация)
- **Gradient clipping**
- **Learning rate scheduling**
  - cosine decay
  - warmup
  - step decay

### AdamW + Cosine LR пример

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class MLP(nn.Module):
    def __init__(self, d_in, d_hid, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = MLP(d_in=300, d_hid=256, n_classes=10)
opt = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
sched = CosineAnnealingLR(opt, T_max=50)

for epoch in range(50):
    for Xb, yb in loader:
        opt.zero_grad()
        logits = model(Xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
    sched.step()
```

## 13) Практикум: от нуля до работающего классификатора

Шаги пайплайна:

1. Нормализовать входы (standardization)
2. Архитектура: `d_in → d_hid → K`, ReLU, dropout
3. Инициализация He (Kaiming) для ReLU
4. Optimizer: **AdamW** + **cosine LR scheduler**
5. Следить за:
   - `train loss`
   - `val loss`
   - `accuracy`
   - **early stopping**
6. Проверить:
   - confusion matrix
   - дисбаланс классов (class imbalance)
     - weighted loss / oversampling / focal loss

### Функция для вычисления accuracy

```python
import torch
from sklearn.metrics import accuracy_score

def evaluate(model, loader):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            logits = model(Xb)
            preds.append(logits.argmax(dim=1).cpu())
            gold.append(yb.cpu())

    preds = torch.cat(preds).numpy()
    gold = torch.cat(gold).numpy()
    return accuracy_score(gold, preds)
```

## 14) Формулы (cheat-sheet)

**Softmax**
\[
\operatorname{softmax}(z)_k \;=\; \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
\]

**Кросс-энтропия (one-hot)**
\[
\mathcal{L}(x,y) \;=\; -\sum_{k=1}^{K} y_k \,\log \operatorname{softmax}(z)_k
\;=\; -\log p_{y^\star}
\]

**Ключевой градиент по логитам**
\[
\frac{\partial \mathcal{L}}{\partial z} \;=\; p - y
\]

**Линейный слой**
\[
z = Wx + b, \qquad
\frac{\partial \mathcal{L}}{\partial W} = (p-y)\,x^{\top}, \qquad
\frac{\partial \mathcal{L}}{\partial b} = p-y, \qquad
\frac{\partial \mathcal{L}}{\partial x} = W^{\top}(p-y)
\]

**Backprop через ReLU**
\[
h = \operatorname{ReLU}(a)=\max(0,a), \quad
\frac{\partial \mathcal{L}}{\partial a} \;=\; \frac{\partial \mathcal{L}}{\partial h}\;\odot\;\mathbf{1}_{a>0}
\]

**Полносвязная сеть (1 скрытый слой)**
\[
\begin{aligned}
h &= f(W_1 x + b_1),\\
z &= W_2 h + b_2,\\
p &= \operatorname{softmax}(z),\\
\mathcal{L} &= -\log p_{y^\star}
\end{aligned}
\]

Градиенты:
\[
\begin{aligned}
g_z &= \frac{\partial \mathcal{L}}{\partial z} = p - y,\\
\frac{\partial \mathcal{L}}{\partial W_2} &= g_z\, h^\top,\qquad
\frac{\partial \mathcal{L}}{\partial b_2} = g_z,\\
g_h &= W_2^\top g_z,\\
\frac{\partial \mathcal{L}}{\partial W_1} &= \big(g_h \odot f'(W_1x+b_1)\big)\,x^\top, \qquad
\frac{\partial \mathcal{L}}{\partial b_1} = g_h \odot f'(W_1x+b_1)
\end{aligned}
\]

**L2-регуляризация (weight decay)**
\[
\mathcal{L}_{\text{total}} \;=\; \mathcal{L}_{\text{task}} \;+\; \frac{\lambda}{2}\,\lVert \theta \rVert_2^2
\]

**Cosine similarity (на всякий случай)**
\[
\cos(\mathbf{a},\mathbf{b}) \;=\; \frac{\mathbf{a}^\top \mathbf{b}}{\lVert \mathbf{a}\rVert_2\,\lVert \mathbf{b}\rVert_2}
\]

## 15) Литература и полезные материалы

**Базовые источники**

- **Rumelhart, Hinton, Williams (1986)**  
  *Learning Representations by Backpropagating Errors*  
  — Классическая статья, где впервые описан backprop для нейросетей.

- **Collobert, Weston et al. (2011)**  
  *Natural Language Processing (Almost) from Scratch*  
  — Первая мощная нейросеточная архитектура для NLP, без ручных признаков.

- **Baydin et al. (2015)**  
  *Automatic Differentiation in Machine Learning: A Survey*  
  — Глубокий обзор методов автоматического дифференцирования (autodiff).

- **Karpathy (2016)**  
  *Yes, you should understand backprop*  
  — Отличное интуитивное объяснение backprop, MUST READ.

---

**Что ещё почитать/посмотреть**

- Goodfellow, Bengio, Courville — *Deep Learning Book*, гл. 6–8.
- CS231n (Stanford): лекции по backprop, softmax, SGD.
- PyTorch Tutorials: *Autograd & Optimization*
- Karpathy’s micrograd — минимальная реализация backprop:
  https://github.com/karpathy/micrograd  

---

**Ключевые takeaway-пункты лекции**

- Softmax + Cross-entropy — стандарт для многоклассовой классификации
- Backprop = просто цепное правило на графе вычислений
- ReLU работает лучше tanh для большинства задач
- Weight decay && dropout → контроль переобучения
- Векторизация > циклы руками
- AdamW + cosine schedule — сильный базовый сетап
- Хорошая инициализация = стабильное обучение

---

**Что знать к следующей лекции**

- Как считается gradient = (p − y)
- Как работает `loss.backward()` в PyTorch
- Что такое hidden layer и зачем нелинейности
- В чём смысл weight decay и dropout
- Почему батчи ускоряют обучение даже на CPU
- Как понять, что модель overfitting/underfitting

---
