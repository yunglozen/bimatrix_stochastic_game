from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm

__all__ = ["promo_game", "StackelbergResult", "stackelberg", "stackelberg_pure",
           "quantile_matrices", "leader_payoff_check",
           "find_nash_lemke_howson", "transform_matrices", "read_matrices"]

_TOL = 1e-9

# --------------------------------------------------------------------------
#  РАВНОВЕСИЕ ПО Штакельбергу.
# --------------------------------------------------------------------------

@dataclass
class StackelbergResult:

    """Результат поиска равновесия по Штакельбергу."""

    leader: int                      # 0 — лидер игрок 1 (строки), 1 — игрок 2 (столбцы)
    x: np.ndarray                    # смешанная стратегия лидера
    follower_response: int           # чистый ответ последователя
    y: np.ndarray                    # стратегия последователя
    leader_payoff: float
    follower_payoff: float
    # лучшие значения лидера для каждого индуцируемого ответа (nan — недостижим)
    payoff_by_response: np.ndarray = field(default_factory=lambda: np.array([]))

    def __str__(self) -> str:
        p = lambda v: "(" + ", ".join(f"{t:.4f}" for t in v) + ")"
        return (f"Лидер: игрок {self.leader + 1}\n"
                f"  стратегия лидера x*      = {p(self.x)}\n"
                f"  ответ последователя y*   = {p(self.y)}"
                f"  (чистая стратегия №{self.follower_response + 1})\n"
                f"  выигрыш лидера           = {self.leader_payoff:.6f}\n"
                f"  выигрыш последователя    = {self.follower_payoff:.6f}")


def _prepare(A, B, leader):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape != B.shape:
        raise ValueError("A и B должны быть матрицами одинаковой размерности m×n "
                         f"(получено {A.shape} и {B.shape}).")
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise ValueError("Матрицы содержат NaN/inf.")
    if leader not in (0, 1):
        raise ValueError("leader должен быть 0 (игрок 1) или 1 (игрок 2).")
    # приводим к виду «лидер — строки»: L, F имеют размер (число стратегий лидера × последователя)
    return (A, B) if leader == 0 else (B.T, A.T)


def stackelberg(A, B, leader: int = 0) -> StackelbergResult:

    """
    Сильное равновесие по Штакельбергу в смешанных стратегиях лидера.

    A, B   — матрицы выигрышей игроков 1 и 2, обе размера m×n
             (A[i,j], B[i,j] — выигрыши при выборе i игроком 1 и j игроком 2);
    leader — 0: лидер игрок 1 (строки), 1: лидер игрок 2 (столбцы).
    """

    L, F = _prepare(A, B, leader)
    m, n = L.shape

    best = None
    by_resp = np.full(n, np.nan)
    for j in range(n):
        c = -L[:, j]
        ks = [k for k in range(n) if k != j]
        A_ub = (F[:, ks] - F[:, [j]]).T if ks else None
        b_ub = np.zeros(len(ks)) if ks else None
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      A_eq=np.ones((1, m)), b_eq=[1.0],
                      bounds=[(0, None)] * m, method="highs")
        if res.status != 0:
            continue
        x = np.clip(res.x, 0, None)
        x /= x.sum()
        val = float(L[:, j] @ x)
        by_resp[j] = val
        if best is None or val > best[0] + _TOL:
            best = (val, j, x)

    val, j, x = best
    y = np.zeros(n); y[j] = 1.0
    return StackelbergResult(leader=leader, x=x, follower_response=j, y=y,
                             leader_payoff=val, follower_payoff=float(F[:, j] @ x),
                             payoff_by_response=by_resp)


def stackelberg_pure(A, B, leader: int = 0):

    """
    Равновесие по Штакельбергу, когда лидер ограничен чистыми стратегиями.
    Возвращает (i, j, выигрыш лидера, выигрыш последователя); тай-брейк — в пользу лидера.
    """

    L, F = _prepare(A, B, leader)
    best = None
    for i in range(L.shape[0]):
        row = F[i]
        br = np.flatnonzero(row >= row.max() - _TOL)
        j = br[np.argmax(L[i, br])]
        if best is None or L[i, j] > best[2] + _TOL:
            best = (i, int(j), float(L[i, j]), float(F[i, j]))
    return best


def leader_payoff_check(A, B, x, leader: int = 0):

    """Проверка: выигрыш лидера при стратегии x и оптимальном (в его пользу) ответе последователя."""

    L, F = _prepare(A, B, leader)
    x = np.asarray(x, float)
    f = x @ F
    br = np.flatnonzero(f >= f.max() - 1e-7)
    return float(max((x @ L)[br]))


def quantile_matrices(mu_A, mu_B, sigma_A, sigma_B, alpha: float = 0.95,
                      cost=None, scale: float = 1000.0):

    """
    Переход от стохастической игры к детерминированной (квантильный критерий):

        a~_ij = (mu_A[i,j] - z_alpha * sigma_A[i,j]) / cost[...] * scale,
        b~_ij = (mu_B[i,j] - z_alpha * sigma_B[i,j]) / cost[...] * scale.

    mu_*    — матрицы мат. ожиданий (m×n);
    sigma_* — матрицы стандартных отклонений либо вектор длины m (σ задаётся по строкам)
              / длины n (по столбцам) — для A, B соответственно см. ниже;
    cost    — None (без нормировки) или матрица m×n той же формы;
    alpha   — уровень доверия, z_alpha = Φ^{-1}(alpha).
    """

    z = norm.ppf(alpha)
    mu_A = np.asarray(mu_A, float); mu_B = np.asarray(mu_B, float)
    sA = np.broadcast_to(np.asarray(sigma_A, float), mu_A.shape)
    sB = np.broadcast_to(np.asarray(sigma_B, float), mu_B.shape)
    A = mu_A - z * sA
    B = mu_B - z * sB
    if cost is not None:
        c = np.broadcast_to(np.asarray(cost, float), mu_A.shape)
        A = A / c * scale
        B = B / c * scale
    return A, B


# --------------------------------------------------------------------------
#  РАВНОВЕСИЕ ПО НЭШУ: алгоритм Лемке–Хоусона
# --------------------------------------------------------------------------


def define_d(A, B):

    """Константа d, большая всех элементов обеих матриц (d = max(A, B) + 1).

    Нужна, чтобы матрицы d - A и d - B были строго положительными —
    техническое условие корректной работы метода Лемке–Хоусона.
    """

    return max(np.max(A), np.max(B)) + 1


def transform(A, d):

    """Преобразование матрицы выигрышей: A1 = d·E - A (E — матрица из единиц)."""

    n, m = A.shape
    E = np.ones((n, m))
    return d * E - A


def y_indx(A_0, indx, e):

    """Начальная вершина: строка `indx` матрицы A_0, её минимальный элемент.

    Возвращает (y, i_x, i_y): вектор y (строка базиса e с заменённой на 1/a
    компонентой), индекс i_x минимального элемента строки и индекс строки i_y.
    Замечание: строка базиса e изменяется на месте (как в исходной реализации).
    """

    i_x = 0
    i_y = 0
    a = min(A_0[indx])
    for i in range(len(A_0[indx])):
        if A_0[indx][i] == a:
            i_x = i
            i_y = indx
            break
    y = e[indx]
    y[indx] = 1 / a
    return y, i_x, i_y


def complication_A1(A_0, y_i, i_x, i_y):

    """Шаг замены базиса (пивотирование) матрицы A_0 по элементу (i_y, i_x).

    Возвращает преобразованную матрицу A_1 и обновлённый базис e.
    """

    n, m = A_0.shape
    e = np.eye(n)
    e[i_y] = y_i
    coef = y_i[i_y]
    A_1 = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if i == i_y:
                A_1[i][j] = A_1[i][j] + A_0[i][j] * y_i[i_y]
            else:
                A_1[i][j] = A_0[i][j] - A_0[i][i_x] * A_0[i_y][j] * coef
    for i in range(n):
        for j in range(i_y, i_y + 1):
            if i != i_y:
                e[i][j] = e[i][j] - A_0[i][i_x] * coef
    return A_1, e


def find_xsi(A_1, y_i):

    """Вектор невязок ξ_i = (A_1 y)_i - 1 для текущей вершины."""

    xsi = []
    for i in range(len(A_1)):
        xsi.append(float(np.dot(A_1[i], y_i)) - 1)
    return xsi


def find_lambda(A_2, new_e, xsi, y_i, letter_1, letter_2):

    """Выбор параметра λ (длины шага вдоль ребра) для каждой строки.

    Для каждой строки i перебираются ограничения по столбцам матрицы A_2
    (метка `letter_1`) и по базису new_e (метка `letter_2`); берётся
    наибольшее положительное либо наименьшее отрицательное отношение.
    Возвращает список пар [λ_i, метка ограничения, определяющего λ_i].
    """

    n, m = A_2.shape
    lambda_mx = []
    lambda_mn = []
    lambda_l = []
    for i in range(n):
        lambda_max = []
        lambda_min = []
        max_coef = []
        min_coef = []
        for j in range(m):
            if A_2[i][j] > 0:
                lambda_max.append(-xsi[j] / A_2[i, j])
                max_coef.append([letter_1, j])
            elif A_2[i][j] < 0:
                lambda_min.append(-xsi[j] / A_2[i, j])
                min_coef.append([letter_1, j])
        for j in range(n):
            if new_e[i][j] > 0:
                lambda_max.append(-y_i[j] / new_e[i, j])
                max_coef.append([letter_2, j])
            elif new_e[i][j] < 0:
                lambda_min.append(-y_i[j] / new_e[i, j])
                min_coef.append([letter_2, j])
        if len(lambda_max) == 0:
            lambda_mx.append([0, 0])
        else:
            lambda_mx.append([max(lambda_max),
                              max_coef[lambda_max.index(max(lambda_max))]])
        if len(lambda_min) == 0:
            lambda_mn.append([0, 0])
        else:
            lambda_mn.append([min(lambda_min),
                              min_coef[lambda_min.index(min(lambda_min))]])

    for i in range(len(lambda_mx)):
        if lambda_mx[i][0] != 0 or lambda_mn[i][0] != 0:
            if lambda_mx[i][0] != 0:
                lambda_l.append(lambda_mx[i])
            else:
                lambda_l.append(lambda_mn[i])
        elif (lambda_mx[i][0] == 0) and (lambda_mn[i][0] == 0):
            lambda_l.append([0, 0])
    return lambda_l


def find_y_T(y_i, lambda_a, new_e):

    """Новые точки y_T[i] = y + λ_i · e_i (шаг вдоль ребра многогранника)."""

    y_T = []
    for i in range(len(lambda_a)):
        y_T.append(y_i + lambda_a[i][0] * new_e[i])
    return y_T


def matrix_e(letter, e):

    """Начальная таблица базисов: P[i][j] = [letter, j] для всех i, j."""

    n, m = e.shape
    P = np.empty((n, n), dtype="object")
    for i in range(n):
        for j in range(n):
            P[i][j] = [letter, j]
    return P


def change_P(P, indx, i_x, letter):

    """Заменяет в столбце `indx` таблицы P все метки на [letter, i_x]."""

    n, m = P.shape
    for i in range(n):
        for j in range(m):
            if j == indx:
                P[i][j] = [letter, i_x]
    return P


def change_P_new(Q, lambda_a):

    """Обновляет диагональ таблицы Q метками, найденными в find_lambda."""

    n, m = Q.shape
    for i in range(n):
        for j in range(m):
            if lambda_a[i][1] != 0:
                Q[i][i] = lambda_a[i][1]
    return Q


def P_and_Q_equilibrium(P, Q):

    """Проверка условия полноты меток: ищет пару вершин (ip, iq), для которых
    объединение меток таблиц P и Q содержит все стратегии обоих игроков.

    Возвращает [ip, iq] (индексы вершин, соответствующих равновесию) либо
    пустой список, если такой пары пока нет.
    """

    res = []
    n, m = P.shape
    n1, m1 = Q.shape
    p_1 = [i for i in range(n)]
    q_1 = [i for i in range(n1)]
    for ip in range(n):
        for iq in range(n1):
            p_0 = []
            q_0 = []
            for jp in range(n):
                if P[ip][jp][0] == 'b':
                    p_0.append(P[ip][jp][1])
                else:
                    q_0.append(P[ip][jp][1])
            for jq in range(n1):
                if Q[iq][jq][0] == 'e':
                    p_0.append(Q[iq][jq][1])
                else:
                    q_0.append(Q[iq][jq][1])
            p_0.sort()
            q_0.sort()
            if p_0 == q_1 and q_0 == p_1:
                res = [ip, iq]
                break
    return res


def read_matrices():

    """Интерактивный ввод матриц A (N×M) и B (M×N) с клавиатуры.

    Размеры запрашиваются у пользователя, строки вводятся через пробел.
    Возвращает пару numpy-массивов (A, B).
    """

    N = int(input("Введите количество строк N для матрицы A: "))
    M = int(input("Введите количество столбцов M для матрицы A: "))

    print(f"\nВведите элементы матрицы A ({N} строк по {M} чисел):")
    A = []
    for i in range(N):
        row = list(map(float, input(f"Строка {i + 1}: ").split()))
        while len(row) != M:
            print(f"ОШИБКА: Введите ровно {M} чисел.")
            row = list(map(float, input(f"Строка {i + 1}: ").split()))
        A.append(row)
    A = np.array(A, dtype=float)

    print(f"\nВведите элементы матрицы B ({M} строк по {N} чисел):")
    B = []
    for i in range(M):
        row = list(map(float, input(f"Строка {i + 1}: ").split()))
        while len(row) != N:
            print(f"ОШИБКА: Введите ровно {N} чисел.")
            row = list(map(float, input(f"Строка {i + 1}: ").split()))
        B.append(row)
    B = np.array(B, dtype=float)

    return A, B


def transform_matrices(A_l, B_l, z, sigma):

    """Переход к детерминированным матрицам по квантильному критерию.

    A_l, B_l — матрицы математических ожиданий (N×M и M×N);
    z        — квантиль стандартного нормального распределения;
    sigma    — вектор длины N + M: первые N значений — σ по строкам для A,
               последние M — σ по столбцам для B.
    Возвращает (A, B) с элементами «мат. ожидание − z·σ».
    (Аналог quantile_matrices для формата ввода из отчёта.)
    """

    A_l = np.array(A_l, dtype=float)
    B_l = np.array(B_l, dtype=float)

    N, M = A_l.shape
    M2, N2 = B_l.shape

    assert M == M2 and N == N2, "Матрицы A и B должны быть размерностей N×M и M×N соответственно."
    assert len(sigma) == N + M, f"Массив sigma должен иметь длину {N + M}"

    # Обработка A: по строкам
    sigma_A = np.array(sigma[:N]).reshape(N, 1)  # (N, 1)
    A = A_l - z * sigma_A

    # Обработка B: по столбцам
    sigma_B = np.array(sigma[N:]).reshape(1, M)  # (1, M)
    B = B_l - z * sigma_B

    return A, B


def find_nash_lemke_howson(A, B):

    """Равновесие Нэша в смешанных стратегиях методом Лемке–Хоусона.

    A, B — матрицы выигрышей игроков 1 и 2.
    Матрицы преобразуются к виду d - A, d - B, затем
    последовательно перебираются начальные вершины и выполняются шаги замены
    базиса, пока таблицы меток P и Q не станут полными
    (P_and_Q_equilibrium).

    Возвращает (x, y) — смешанные стратегии игроков 1 и 2 (numpy-векторы,
    сумма компонент равна 1). Если равновесие найти не удалось, возвращает
    None.
    """

    n, m = A.shape

    d = define_d(A, B)

    A_1 = transform(A, d)
    B_1 = transform(B, d)

    A_0 = A_1.transpose()
    e = np.eye(m)

    B_0 = B_1
    f = np.eye(n)

    P = matrix_e('f', f)
    Q = matrix_e('e', e)

    x = y = None
    for indx in range(len(A)):
        y_i, i_x, i_y = y_indx(A_0, indx, e)
        x_i, j_x, j_y = y_indx(B_0, i_x, f)
        Q = change_P(Q, indx, i_x, 'a')
        P = change_P(P, i_x, j_x, 'b')

        if len(P_and_Q_equilibrium(P, Q)) == 0:
            A_2, new_e = complication_A1(A_0, y_i, i_x, i_y)
            B_2, new_f = complication_A1(B_0, x_i, j_x, j_y)

            xsi = find_xsi(A_1, y_i)
            psi = find_xsi(B_1.transpose(), x_i)

            lambda_a = find_lambda(A_2, new_e, xsi, y_i, 'a', 'e')
            lambda_b = find_lambda(B_2, new_f, psi, x_i, 'b', 'f')

            y_T = find_y_T(y_i, lambda_a, new_e)
            x_T = find_y_T(x_i, lambda_b, new_f)

            P = change_P_new(P, lambda_b)
            Q = change_P_new(Q, lambda_a)

            res = P_and_Q_equilibrium(P, Q)
            if len(res) != 0:
                x = x_T[res[0]] / sum(x_T[res[0]])
                y = y_T[res[1]] / sum(y_T[res[1]])
                break
        else:
            return None

    if x is None or y is None:
        return None
    return x, y


# --------------------------------------------------------------------------
#  Игра: конкуренция двух сетей фитнес-клубов (стратегии 1..n, эффект делится
#  пополам при совпадении стратегий)
# --------------------------------------------------------------------------
def promo_game(cost, reach_A, reach_B, sigma, alpha=0.95):

    """
    Строит матрицы игры «конкуренция сетей» из отчёта.
    cost, reach_A, reach_B — векторы длины n; sigma — вектор длины 2n
    (первые n — σ для игрока A по строкам, последние n — σ для B по столбцам).
    """

    cost = np.asarray(cost, float); rA = np.asarray(reach_A, float); rB = np.asarray(reach_B, float)
    sigma = np.asarray(sigma, float)
    n = len(cost)
    assert len(rA) == len(rB) == n and len(sigma) == 2 * n
    muA = np.tile(rA[:, None], (1, n)); muB = np.tile(rB[None, :], (n, 1))
    np.fill_diagonal(muA, rA / 2); np.fill_diagonal(muB, rB / 2)
    sA = np.tile(sigma[:n, None], (1, n)); sB = np.tile(sigma[None, n:], (n, 1))
    cA = np.tile(cost[:, None], (1, n)); cB = np.tile(cost[None, :], (n, 1))
    z = norm.ppf(alpha)
    A = (muA - z * sA) / cA * 1000.0
    B = (muB - z * sB) / cB * 1000.0
    return A, B


if __name__ == "__main__":

    # Пример: фитнес-клубы 6×6.
    names = ["Скидки на абонемент", "Бесплатное пробное занятие", "Реклама в соц. сетях",
             "Партнёрство с работодателями", "Новая групповая программа",
             "Программа лояльности"]
    A, B = promo_game(cost=[750, 500, 700, 850, 650, 800],
                      reach_A=[95, 60, 100, 115, 85, 105],
                      reach_B=[90, 55, 95, 110, 80, 100],
                      sigma=list(range(1, 13)), alpha=0.95)
    np.set_printoptions(precision=2, suppress=True, linewidth=140)
    print("A =\n", A, "\nB =\n", B, "\n")

    print("=== Равновесие по Штакельбергу ===")

    for lead in (0, 1):
        r = stackelberg(A, B, leader=lead)
        print(r)
        print(f"  ответ: «{names[r.follower_response]}»")
        print("  выигрыши по индуцируемым ответам:", r.payoff_by_response, "\n")

    print("=== Равновесие Нэша (метод Лемке–Хоусона) ===")

    nash = find_nash_lemke_howson(A, B)
    if nash is None:
        print("Равновесие не найдено.")
    else:
        x, y = nash
        print("x* =", np.round(x, 4), "\ny* =", np.round(y, 4))
        print(f"выигрыши: игрок 1 = {x @ A @ y:.4f}, игрок 2 = {x @ B @ y:.4f}")
