---
layout: default
title: Chapter 1 — Stable Matching & Five Problems
---

# Chapter 1 — Stable Matching & Five Representative Problems  
**Exam-ready, bilingual summary (EN ↔ RU) with proofs, algorithms, and key takeaways**  
_Figures referenced: 1.1–1.7._

---

## Contents
1. [1.1 Stable Matching / Устойчивое совпадение](#11-stable-matching--устойчивое-совпадение)  
   1. [Motivation / Мотивация](#motivation--мотивация)  
   2. [Definitions / Определения](#definitions--определения)  
   3. [Gale–Shapley Algorithm / Алгоритм Гейла–Шепли](#gale–shapley-algorithm--алгоритм-гейла–шепли)  
   4. [Correctness & Bounds / Корректность и оценки](#correctness--bounds--корректность-и-оценки)  
   5. [Optimality & “Unfairness” / Оптимальность и «нечестность»](#optimality--“unfairness”--оптимальность-и-«нечестность»)  
   6. [Small Examples / Малые примеры](#small-examples--малые-примеры)  
2. [1.2 Five Representative Problems / Пять показательных задач](#12-five-representative-problems--пять-показательных-задач)  
   1. [Graph Basics / База по графам](#graph-basics--база-по-графам)  
   2. [Interval Scheduling (Greedy) / Планирование интервалов (жадно)](#interval-scheduling-greedy--планирование-интервалов-жадно)  
   3. [Weighted Interval Scheduling (DP) / Взвешенные интервалы (ДП)](#weighted-interval-scheduling-dp--взвешенные-интервалы-дп)  
   4. [Bipartite Matching (Augmentation/Flow) / Двудольное паросочетание](#bipartite-matching-augmentationflow--двудольное-паросочетание)  
   5. [Independent Set (NP-complete) / Независимое множество](#independent-set-np-complete--независимое-множество)  
   6. [Competitive Facility Location (PSPACE) / Конкурентное размещение (PSPACE)](#competitive-facility-location-pspace--конкурентное-размещение-pspace)  
3. [Cheat Sheet / Шпаргалка](#cheat-sheet--шпаргалка)

---

## 1.1 Stable Matching / Устойчивое совпадение

### Motivation / Мотивация
**EN.** Admissions/recruiting can “unravel”: after acceptances, a company–applicant pair that prefers each other can defect, cascading changes. We want an assignment where **no such pair exists**—then self-interest enforces the outcome. (Real systems: NRMP.)  
**RU.** Процессы приёма/найма могут «разматываться»: пара работодатель–кандидат, взаимно предпочитающая друг друга, нарушает статус-кво. Нужна раздача, где **нет таких пар** — тогда самоинтерес удерживает всех. (Практика: NRMP.)

### Definitions / Определения
**EN**
- Sets of equal size: men \(M\), women \(W\). Strict preference lists over the opposite set.
- **Matching:** subset of \(M\times W\) with no person in more than one pair.
- **Perfect matching:** everyone matched.
- **Instability (blocking pair)** (Fig. 1.1): \((m,w')\notin S\) where \(m\) prefers \(w'\) to his partner in \(S\), and \(w'\) prefers \(m\) to hers.
- **Stable matching:** perfect + no blocking pair.

![Figure 1.1 – Instability example]({{ '/images/fig1_1.png' | relative_url }})

**RU**
- Множества одинакового размера: мужчины \(M\), женщины \(W\). Строгие предпочтения.
- **Совпадение:** пары из \(M\times W\) без повторов людей.
- **Совершенное совпадение:** все состоят в парах.
- **Нестабильность (блокирующая пара):** \((m,w')\notin S\), где оба предпочитают друг друга текущим партнёрам.
- **Устойчивое совпадение:** совершенное + без блокирующих пар.

### Gale–Shapley Algorithm / Алгоритм Гейла–Шепли
**EN (men-proposing, “deferred acceptance”)**
1. While some man \(m\) is free and has untried women, he proposes to the highest-ranked unproposed \(w\).
2. If \(w\) is free → engage \((m,w)\).  
   Else \(w\) compares \(m\) with current fiancé \(m'\): keeps her preferred; rejects the other (Fig. 1.2).
3. When no man is free → finalize engagements; return \(S\).

**RU (мужчины делают предложения)**
1. Пока есть свободный \(m\) с непросмотренными женщинами, он предлагает лучшей из ещё не предложенных \(w\).
2. Если \(w\) свободна — помолвка; иначе сравнивает с \(m'\), оставляет лучшего, другого отвергает.
3. Когда свободных мужчин нет — фиксация помолвок; результат \(S\).

**Key invariants / Инварианты**
- **Women:** once first proposal arrives, always engaged; partners only improve.  
- **Men:** propose in strictly worsening order down their lists.

### Correctness & Bounds / Корректность и оценки
**EN**
- **Termination:** at most \(n^2\) proposals (each ordered pair proposed once) ⇒ \(O(n^2)\) time.  
- **Perfection:** assume a free man proposed to all → then all women are engaged ⇒ \(n\) engaged men, contradiction.  
- **Stability:** if \((m,w')\) blocked the output, \(m\) must have proposed to \(w'\) earlier and was rejected for someone she prefers; she never ends with someone worse ⇒ contradiction.

**RU**
- **Окончание:** не более \(n^2\) предложений ⇒ время \(O(n^2)\).  
- **Совершенство:** свободный мужчина, предложивший всем, невозможен (иначе число женихов \(=n\)).  
- **Устойчивость:** блокирующая пара невозможна, т.к. женщина уже отвергла этого мужчину ради более предпочтительного и дальше только улучшает партнёров.

### Optimality & “Unfairness” / Оптимальность и «нечестность»
**EN**
- **Valid partner:** \(w\) is valid for \(m\) if some stable matching contains \((m,w)\).  
- **Men-optimal matching \(S^\*\):** pairs \((m,\text{best}(m))\) where \(\text{best}(m)\) is \(m\)’s best valid partner.  
- **Theorem:** Every execution of men-proposing GS returns **the same** \(S^\*\) (order of proposals irrelevant).  
  *Idea:* if a man were ever rejected by his best valid partner, combine with another stable matching to produce a contradiction (a blocking pair).  
- **Dual:** In \(S^\*\), each woman gets her **worst** valid partner. If women propose, roles reverse (women-optimal, men-pessimal).
- **Multiplicity:** multiple stable matchings may exist; GS picks the proposer-optimal one.

**RU**
- **Допустимый партнёр:** присутствует с тобой в некотором устойчивом совпадении.  
- **Мужчинам-оптимальное \(S^\*\):** каждый мужчина получает своего **лучшего** допустимого партнёра; все запуски выдают один и тот же результат.  
- **Дуальность:** каждая женщина — своего **худшего** допустимого; при предложениях женщин — наоборот.  
- **Множественность:** устойчивых совпадений может быть несколько; GS выбирает оптимальное для стороны-инициатора.

### Small Examples / Малые примеры
**EN**
1. **Total agreement:** both men rank \(w \succ w'\), both women rank \(m \succ m'\) ⇒ unique stable matching \(\{(m,w),(m',w')\}\).  
2. **Clashing preferences:** men prefer different first choices; women prefer the opposite arrangement ⇒ two stable matchings: men-optimal and women-optimal.

**RU**
1. **Полное согласие:** единственное устойчивое совпадение — «старшие со старшими».  
2. **Конфликтующие предпочтения:** две устойчивые раздачи — оптимальная для мужчин и для женщин.

---

## 1.2 Five Representative Problems / Пять показательных задач

### Graph Basics / База по графам
**EN.** Graph \(G=(V,E)\): vertices/nodes \(V\), edges \(E\subseteq \{\{u,v\}\}\). Visualized as circles and segments (Fig. 1.3).  
**RU.** Граф \(G=(V,E)\): вершины и рёбра; рисуются как окружности и линии (рис. 1.3).

---

### Interval Scheduling (Greedy) / Планирование интервалов (жадно)
**EN**
- **Input:** intervals \((s_i,f_i)\). Two intervals compatible if they don’t overlap.  
- **Goal:** maximize number of intervals (Fig. 1.4).  
- **Greedy rule (optimal):** sort by **earliest finish time**; scan once, keep interval if it starts ≥ current finish.  
- **Why it works (exchange idea):** in any optimal solution, the earliest-finishing interval among remaining can be chosen without reducing the final count; swapping moves finish time leftward, never hurting feasibility.  
- **Complexity:** sorting \(O(n\log n)\), scan \(O(n)\).

**RU**
- **Вход:** интервалы \((s_i,f_i)\); совместимы, если не пересекаются.  
- **Цель:** максимальное число интервалов.  
- **Жадное правило:** сортировать по раннему окончанию; добавлять совместимые.  
- **Оптимальность (обмен):** самый ранний по окончанию можно всегда «вытолкнуть» в оптимальное решение без потери; заменой улучшаем край без ущерба размеру.  
- **Сложность:** \(O(n\log n)\) на сортировку + \(O(n)\) проход.

---

### Weighted Interval Scheduling (DP) / Взвешенные интервалы (ДП)
**EN**
- **Input:** intervals with values \(v_i>0\).  
- **Goal:** maximize total value of compatible set.  
- **DP setup:** sort by \(f_i\). For interval \(i\), let \(p(i)\) be the rightmost index \(<i\) with \(f_{p(i)}\le s_i\) (compute via binary search).  
- **Recurrence:**  
  \[
  OPT(i)=\max\{\,v_i+OPT(p(i)),\; OPT(i-1)\,\},\quad OPT(0)=0.
  \]
- **Recover solution:** traceback: if \(v_i+OPT(p(i))>OPT(i-1)\) include \(i\) and jump to \(p(i)\), else skip \(i\).  
- **Complexity:** compute \(p(i)\) in \(O(\log n)\) each ⇒ \(O(n\log n)\); if \(p\) precomputed linearly on a timeline, \(O(n)\).

**RU**
- **Вход:** интервалы с ценностями \(v_i\).  
- **Цель:** максимизировать суммарную ценность.  
- **ДП:** сортировка по \(f_i\); функция \(p(i)\) — последний непересекающийся слева.  
- **Рекуррент:** \(OPT(i)=\max\{v_i+OPT(p(i)),\,OPT(i-1)\}\), \(OPT(0)=0\).  
- **Восстановление:** обратный проход; включаем \(i\), если выгодно.  
- **Сложность:** \(O(n\log n)\) (или \(O(n)\) при линейном вычислении \(p\)).

---

### Bipartite Matching (Augmentation/Flow) / Двудольное паросочетание
**EN**
- **Bipartite graph:** \(V=X\cup Y\), edges only between \(X\) and \(Y\) (Fig. 1.5).  
- **Matching:** edges sharing no endpoints; **perfect** if \(|M|=|X|=|Y|\).  
- **Goal:** maximum matching.  
- **Augmenting path idea:** a path that alternates unmatched/matched edges, starting and ending at free vertices; flipping along it increases matching size by 1.  
- **Algorithms:**  
  - Simple DFS-based augmentation: \(O(VE)\).  
  - **Hopcroft–Karp:** repeatedly finds many shortest augmenting paths; \(O(E\sqrt{V})\).  
  - Flow reduction: convert to s–t network with unit capacities.  
- **Uses:** assignment (jobs↔machines), timetabling, pairing.

**RU**
- **Двудольный граф:** \(V=X\cup Y\), рёбра только между долями.  
- **Паросочетание:** рёбра без общих концов; совершенное — покрывает все вершины.  
- **Цель:** максимальный размер.  
- **Идея увеличивающих путей:** чередующийся путь от свободной вершины к свободной; инвертируем принадлежность рёбер — размер растёт.  
- **Алгоритмы:** DFS \(O(VE)\), Хопкрофт–Карп \(O(E\sqrt{V})\), или через сетевой поток.  
- **Применения:** назначение работ, расписания, покрытия.

---

### Independent Set (NP-complete) / Независимое множество
**EN**
- **Definition:** \(S\subseteq V\) is independent if no two vertices in \(S\) are adjacent (Fig. 1.6).  
- **Goal:** maximize \(|S|\).  
- **Expressiveness:** models “choose many items with pairwise conflicts.”  
- **Encodings:**  
  - Interval Scheduling ⇒ conflict graph (edge = overlap).  
  - Matching ⇒ line graph of a bipartite graph (nodes = edges; conflicts share an endpoint).  
- **Complexity:** **NP-complete**—no known poly-time algorithm; verifying a proposed \(S\) is easy (polynomial).  
- **Exam tip:** contrast “find vs verify”: easy verification does not imply easy computation.

**RU**
- **Определение:** независимое множество — вершины без смежных пар.  
- **Цель:** максимальный размер.  
- **Выразительность:** «выбрать много несовместимых объектов».  
- **Кодировки:** интервалы → граф конфликтов; паросочетание → линейный граф.  
- **Сложность:** **NP-полная**; проверка решения проста, поиск — труден.

---

### Competitive Facility Location (PSPACE) / Конкурентное размещение (PSPACE)
**EN**
- **Game:** players P1 and P2 alternately pick vertices (zones) with values \(b_i\); union must remain an independent set (adjacent zones forbidden). P2 wants total ≥ \(B\) (Fig. 1.7).  
- **Hardness:** **PSPACE-complete**—determining existence of a winning strategy is believed harder than NP-complete; even *certifying* a win may require exploring an exponential game tree.  
- **Perspective:** classic adversarial planning / minimax search setting.

**RU**
- **Игра:** игроки по очереди выбирают вершины с весами; выбранные должны оставаться независимыми; цель P2 — сумма ≥ \(B\).  
- **Сложность:** **PSPACE-полная** — решение и даже проверка стратегии требуют анализа дерева игры.  
- **Контекст:** моделирует конкурентное планирование и игры разума.

---

## Cheat Sheet / Шпаргалка

**Definitions to state cleanly / Чёткие определения**
- Matching / Perfect / Blocking pair / Stable  
- Bipartite graph / Matching / Augmenting path  
- Independent set  
- Complexity classes: **P**, **NP-complete**, **PSPACE-complete**

**Algorithms to recall / Алгоритмы**
- **Gale–Shapley** (men-proposing):  
  - ≤ \(n^2\) proposals; perfect + stable.  
  - Men get best valid partners; women worst valid (roles flip if women propose).  
- **Interval Scheduling (greedy by earliest finish)**: optimal; \(O(n\log n)\).  
- **Weighted Intervals (DP)**: \(OPT(i)=\max\{v_i+OPT(p(i)), OPT(i-1)\}\); traceback; \(O(n\log n)\).  
- **Bipartite Matching**: augmentation; Hopcroft–Karp \(O(E\sqrt{V})\) or max-flow.

**Proof nuggets / Идеи доказательств**
- GS stability: rejected once ⇒ never ends worse; contradiction for blocking pair.  
- Greedy intervals: exchange argument with earliest finisher.  
- DP intervals: optimal substructure via \(p(i)\).  
- Matching optimality: existence and use of augmenting paths.

**Conceptual contrasts / Контрасты**
- **Proposer-optimal vs receiver-pessimal** in GS.  
- **Find vs Verify:** Independent Set easy to verify, hard to find; Competitive Facility Location hard even to verify succinctly.

---
