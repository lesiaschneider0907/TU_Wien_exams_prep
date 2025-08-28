---
layout: default
title: Chapter 1 — Stable Matching & Five Representative Problems
---

# Chapter 1 — Stable Matching & Five Representative Problems  
**Exam-ready, bilingual summary (EN ↔ RU) with proofs, algorithms, key takeaways, and figures.**

---

## 1.1 Stable Matching / Устойчивое совпадение

### Definitions / Определения

**Instability (blocking pair):** $(m,w')\notin S$ where both prefer each other to current partners.

![Figure 1.1 – Instability in matching]({{ '/images/fig1_1.png' | relative_url }})  
*Figure 1.1 – Perfect matching with instability (m,w’).*

---

### Gale–Shapley Algorithm / Алгоритм Гейла–Шепли

**EN (men-proposing, “deferred acceptance”)**
1. While some man $m$ is free and has untried women, he proposes to the highest-ranked unproposed $w$.
2. If $w$ is free → engage $(m,w)$.  
   Else $w$ compares $m$ with current fiancé $m'$: keeps her preferred; rejects the other.

**RU (мужчины делают предложения)**
1. Пока есть свободный $m$ с непросмотренными женщинами — предлагает лучшей из ещё не предложенных $w$.
2. Если $w$ свободна — помолвка; иначе $w$ оставляет более предпочтительного из $\{m,m'\}$.

![Figure 1.2 – Intermediate GS state]({{ '/images/fig1_2.png' | relative_url }})  
*Figure 1.2 – State of G-S algorithm when a free man proposes to a woman.*

**Key invariants / Инварианты**
- **Women:** once first proposal arrives, always engaged; partners only improve.  
- **Men:** propose in strictly worsening order down their lists.

**Bounds & Correctness / Оценки и корректность**
- **Termination:** at most $n^2$ proposals ⇒ $O(n^2)$.  
- **Perfection:** a free man who proposed to everyone невозможен (иначе было бы $n$ женихов при $n$ женщинах).  
- **Stability:** if $(m,w')$ blocked the output, then $m$ earlier proposed to $w'$ and was rejected for someone she prefers; she never ends worse ⇒ contradiction.

**Optimality & “Unfairness” / Оптимальность и «нечестность»**
- **Valid partner:** $w$ is valid for $m$ if some stable matching contains $(m,w)$.  
- **Men-optimal $S^\*$:** each man gets $\operatorname{best}(m)$ — his best valid partner; all executions of men-proposing GS return the **same** $S^\*$.  
- **Dual:** each woman gets her **worst** valid partner in $S^\*$; если предлагают женщины — роли меняются.  
- **Multiplicity:** multiple stable matchings can exist (men-optimal vs women-optimal).

---

## 1.2 Five Representative Problems / Пять показательных задач

### Graph Basics / База по графам

![Figure 1.3 – Graphs on four nodes]({{ '/images/fig1_3.png' | relative_url }})  
*Figure 1.3 – Graphs (a) and (b) with four nodes.*

---

### Interval Scheduling (Greedy) / Планирование интервалов (жадно)

![Figure 1.4 – Interval Scheduling instance]({{ '/images/fig1_4.png' | relative_url }})  
*Figure 1.4 – An instance of the Interval Scheduling Problem.*

**EN**
- **Input:** intervals $(s_i,f_i)$. Two intervals are compatible if they don’t overlap.  
- **Goal:** maximize count of intervals.  
- **Greedy (optimal):** sort by earliest finish $f_i$, then take the next compatible.  
- **Why:** exchange argument — earliest finisher can replace the first interval of an optimal solution without reducing its size.  
- **Time:** sort $O(n\log n)$ + scan $O(n)$.

**RU**
- **Жадный алгоритм:** сортировка по раннему окончанию; берём совместимые. Оптимальность — через обменный аргумент.

---

### Weighted Interval Scheduling (DP) / Взвешенные интервалы (ДП)

**EN**
- **Input:** $(s_i,f_i,v_i)$ with $v_i>0$.  
- Sort by $f_i$; let $p(i)$ be the rightmost index $<i$ with $f_{p(i)}\le s_i$.  
- **DP recurrence:**
  $$
  \mathrm{OPT}(i) = \max\{\, v_i + \mathrm{OPT}(p(i)),\; \mathrm{OPT}(i-1) \,\},\quad \mathrm{OPT}(0)=0.
  $$
- **Recover:** traceback (include $i$ if left branch chosen).  
- **Time:** $O(n\log n)$ with binary-searched $p(i)$.

**RU**
- **ДП:** $OPT(i)=\max\{v_i+OPT(p(i)),\,OPT(i-1)\}$; восстановление — обратным проходом.

---

### Bipartite Matching (Augmentation/Flow) / Двудольное паросочетание

![Figure 1.5 – Bipartite graph]({{ '/images/fig1_5.png' | relative_url }})  
*Figure 1.5 – A bipartite graph.*

**Core ideas / Основы**
- Bipartite $G=(X\cup Y,E)$; matching — рёбра без общих концов; perfect — покрывает все вершины.  
- **Augmenting path:** alternating path starting/ending at free vertices; flipping increases $|M|$ by 1.  
- **Algorithms:** DFS-augmentation $O(VE)$; **Hopcroft–Karp** $O(E\sqrt V)$; или через сетевой поток.

---

### Independent Set (NP-complete) / Независимое множество (NP-полная)

![Figure 1.6 – Graph with maximum independent set]({{ '/images/fig1_6.png' | relative_url }})  
*Figure 1.6 – A graph with independent set of size 4.*

- **Definition:** $S\subseteq V$ independent ⇔ no two adjacent.  
- **Power:** models “choose many with pairwise conflicts”.  
- **Encodings:** intervals → conflict graph; matching → line graph.  
- **Complexity:** NP-complete — finding is hard; verifying a proposed $S$ is easy.

---

### Competitive Facility Location (PSPACE) / Конкурентное размещение (PSPACE)

![Figure 1.7 – Facility location instance]({{ '/images/fig1_7.png' | relative_url }})  
*Figure 1.7 – Instance of the Competitive Facility Location Problem.*

- Two players alternate picking weighted vertices; chosen set must stay independent; P2 targets sum $\ge B$.  
- **Complexity:** PSPACE-complete — strategy existence is harder than NP-complete; certification may require exploring an exponential game tree.

---

## Cheat Sheet / Шпаргалка

**Algorithms**
- **Gale–Shapley:** $\le n^2$ proposals; perfect + stable; men-optimal (or women-optimal in the flipped version).
- **Interval Scheduling:** greedy by earliest finish, $O(n\log n)$.  
- **Weighted Intervals:** DP with $p(i)$, $OPT(i)$ above.  
- **Bipartite Matching:** augmenting paths; Hopcroft–Karp $O(E\sqrt V)$.

**Definitions**
- Matching / Perfect / Blocking pair / Stable  
- Bipartite graph / Augmenting path  
- Independent set; Complexity classes P / NP-complete / PSPACE-complete.

**Proof sketches**
- GS stability via “once rejected, always worse for her”.  
- Greedy intervals via exchange.  
- DP via optimal substructure.  
- Matching maximality via augmenting paths.
