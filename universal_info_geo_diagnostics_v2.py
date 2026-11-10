"""
汎用・情報幾何学的診断フレームワーク v2
UniversalInformationGeometricDiagnosticsV2

v1 からの改良点（レビュー指摘への対応）:
  1. スケーリング不変な診断の併記
     - 生の FIM に加えて、相関化 FIM（F_ij / sqrt(F_ii F_jj)）のスペクトル指標を返す。
       相関化スペクトルはパラメータの対角スケーリング（単位変更）に対して不変。
  2. 数値微分の相対ステップ化と中心差分
     - ステップ幅 h_j = eps_rel * max(|theta_j|, scale_j) とし、パラメータの
       大きさに追従させる。デフォルトは中心差分（誤差 O(h^2)）。
       eps_rel の自動既定値: 中心差分 eps_machine^(1/3) ≒ 6.1e-6、
       前方差分 sqrt(eps_machine) ≒ 1.5e-8。
  3. 逆行列のフォールバックを「例外」から「条件数判定」へ
     - np.linalg.inv は近特異では例外を出さず巨大な逆行列を返すため、
       V_reg の条件数を先に評価し、閾値超過時は pinv に切り替える。
       また F の組み立ては明示的な逆行列ではなく線形ソルブ経由を基本とする。
  4. 相対リッジ正則化
     - リッジ量を lambda_rel * (tr(V)/dim) とし、データの代表スケールに比例させる。
  5. 数値ランクの併記
     - 条件数はガード項で飽和するため、lambda_i / lambda_1 > rank_tol の
       本数（数値ランク）と log10(条件数) を併せて返す。
  6. 前提の明示
     - F = J^T Sigma^{-1} J は「Sigma は theta に依存しない既知のノイズ共分散」
       というガウス近似。Sigma を直接渡す経路（covariance=）を用意し、
       標本共分散を使う場合はサンプル数 <= 観測次元で特異になる旨を警告する。
       cov_func（theta -> Sigma）を渡し include_cov_term=True とすると、
       ガウス FIM の共分散依存項  (1/2) Tr(Sigma^{-1} dSigma Sigma^{-1} dSigma)
       も数値微分で加算する。

前提・限界:
  - 局所診断である（評価点 theta の近傍での線形化に基づく）。
  - 確率的シミュレータでは差分がノイズに埋もれる。乱数シード固定
    （common random numbers）等は利用者側の責務とする。
"""

import warnings
from typing import Callable, Dict, Any, Optional

import numpy as np


class UniversalInformationGeometricDiagnosticsV2:
    """
    ブラックボックス・シミュレータ向け情報幾何学的診断エンジン v2。

    Parameters:
    -----------
    model_func : Callable[[np.ndarray], np.ndarray]
        パラメータベクトル theta を受け取り、観測空間の予測値ベクトルを返す関数。
    epsilon : float, optional
        相対ステップの基準 eps_rel。None の場合、差分方式から自動設定する。
    diff_scheme : {"central", "forward"}
        数値微分方式。デフォルト "central"（誤差 O(h^2)、モデル評価 2N 回）。
    param_scales : np.ndarray, optional
        パラメータごとの代表スケール。ステップ幅は
        h_j = eps_rel * max(|theta_j|, scale_j) で決める。
        scale_j 未指定かつ theta_j = 0 の成分では h_j = eps_rel とする。
    """

    _EPS_MACHINE = np.finfo(float).eps

    def __init__(
        self,
        model_func: Callable[[np.ndarray], np.ndarray],
        epsilon: Optional[float] = None,
        diff_scheme: str = "central",
        param_scales: Optional[np.ndarray] = None,
    ):
        if diff_scheme not in ("central", "forward"):
            raise ValueError("diff_scheme は 'central' か 'forward'")
        self.model_func = model_func
        self.diff_scheme = diff_scheme
        if epsilon is None:
            # 最適相対ステップの目安: 中心差分 eps^(1/3), 前方差分 sqrt(eps)
            epsilon = self._EPS_MACHINE ** (1.0 / 3.0) if diff_scheme == "central" \
                else np.sqrt(self._EPS_MACHINE)
        self.epsilon = float(epsilon)
        self.param_scales = None if param_scales is None \
            else np.asarray(param_scales, dtype=float)

    # ------------------------------------------------------------------
    # 数値微分
    # ------------------------------------------------------------------
    def _step(self, theta: np.ndarray, j: int) -> float:
        """パラメータ j に対する相対ステップ幅 h_j。"""
        scale = 0.0
        if self.param_scales is not None:
            scale = abs(float(self.param_scales[j]))
        ref = max(abs(float(theta[j])), scale)
        # theta_j も scale も 0 の場合は eps_rel 自体を絶対ステップにする
        return self.epsilon * ref if ref > 0.0 else self.epsilon

    def compute_jacobian(
        self,
        theta: np.ndarray,
        base_output: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """相対ステップによる数値微分ヤコビアン J ∈ R^{M×N}。"""
        theta = np.asarray(theta, dtype=float)
        n_params = len(theta)

        if self.diff_scheme == "central":
            cols = []
            for j in range(n_params):
                h = self._step(theta, j)
                tp = theta.copy(); tp[j] += h
                tm = theta.copy(); tm[j] -= h
                cols.append((self.model_func(tp) - self.model_func(tm)) / (2.0 * h))
            return np.column_stack(cols)

        # forward
        f0 = base_output if base_output is not None else self.model_func(theta)
        jacobian = np.zeros((len(f0), n_params))
        for j in range(n_params):
            h = self._step(theta, j)
            tp = theta.copy(); tp[j] += h
            jacobian[:, j] = (self.model_func(tp) - f0) / h
        return jacobian

    # ------------------------------------------------------------------
    # 共分散の取得と安定化逆作用素
    # ------------------------------------------------------------------
    def _resolve_covariance(
        self,
        theta: np.ndarray,
        observed_data: Optional[np.ndarray],
        covariance: Optional[np.ndarray],
        cov_func: Optional[Callable[[np.ndarray], np.ndarray]],
    ) -> np.ndarray:
        if covariance is not None:
            cov = np.asarray(covariance, dtype=float)
            if cov.ndim == 0:
                cov = cov.reshape(1, 1)
            return cov
        if cov_func is not None:
            cov = np.asarray(cov_func(theta), dtype=float)
            if cov.ndim == 0:
                cov = cov.reshape(1, 1)
            return cov
        if observed_data is None:
            raise ValueError("covariance / cov_func / observed_data のいずれかが必要です")
        data = np.asarray(observed_data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_samples, n_obs = data.shape
        if n_samples <= n_obs:
            warnings.warn(
                f"標本共分散が特異になります: サンプル数 {n_samples} <= 観測次元 {n_obs}。"
                "既知のノイズ共分散を covariance= で渡すことを推奨します。",
                RuntimeWarning,
            )
        cov = np.cov(data, rowvar=False)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        return cov

    def _stabilized_solve(
        self,
        cov: np.ndarray,
        rhs: np.ndarray,
        ridge_rel: float,
        cond_threshold: float,
    ):
        """
        (V + 相対リッジ) に対する線形ソルブ。条件数が閾値を超えたら pinv 経由。
        戻り値: (解, 使用経路 "solve"/"pinv", V_reg の条件数)
        """
        dim = cov.shape[0]
        scale = np.trace(cov) / dim
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.mean(np.abs(np.diag(cov)))) or 1.0
        cov_reg = cov + np.eye(dim) * (ridge_rel * scale)

        cond = np.linalg.cond(cov_reg)
        if np.isfinite(cond) and cond <= cond_threshold:
            try:
                return np.linalg.solve(cov_reg, rhs), "solve", float(cond)
            except np.linalg.LinAlgError:
                pass
        # 条件数超過 or ソルブ失敗 → 擬似逆行列フォールバック
        inv = np.linalg.pinv(cov_reg, rcond=1e-12)
        return inv @ rhs, "pinv", float(cond)

    # ------------------------------------------------------------------
    # FIM
    # ------------------------------------------------------------------
    def compute_fisher_information_matrix(
        self,
        theta: np.ndarray,
        observed_data: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
        cov_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        ridge_rel: float = 1e-10,
        cond_threshold: float = 1e12,
        include_cov_term: bool = False,
    ):
        """
        F = J^T Sigma^{-1} J （+ 任意で共分散依存項）。

        戻り値: (FIM, メタ情報 dict)
        """
        theta = np.asarray(theta, dtype=float)
        f0 = self.model_func(theta)
        J = self.compute_jacobian(theta, base_output=f0)
        cov = self._resolve_covariance(theta, observed_data, covariance, cov_func)

        solved, path, cov_cond = self._stabilized_solve(cov, J, ridge_rel, cond_threshold)
        fim = J.T @ solved
        meta: Dict[str, Any] = {
            "InversePath": path,
            "CovConditionNumber": cov_cond,
            "CovarianceSource": (
                "given" if covariance is not None else
                "cov_func" if cov_func is not None else "sample"
            ),
        }

        if include_cov_term:
            if cov_func is None:
                raise ValueError("include_cov_term=True には cov_func が必要です")
            # F_ij += (1/2) Tr(Sigma^{-1} d_i Sigma Sigma^{-1} d_j Sigma)
            dim = cov.shape[0]
            eye = np.eye(dim)
            solved_eye, _, _ = self._stabilized_solve(cov, eye, ridge_rel, cond_threshold)
            inv_cov = 0.5 * (solved_eye + solved_eye.T)
            d_covs = []
            for j in range(len(theta)):
                h = self._step(theta, j)
                tp = theta.copy(); tp[j] += h
                tm = theta.copy(); tm[j] -= h
                d_covs.append((np.asarray(cov_func(tp), dtype=float)
                               - np.asarray(cov_func(tm), dtype=float)) / (2.0 * h))
            term = np.zeros_like(fim)
            for i in range(len(theta)):
                ai = inv_cov @ d_covs[i]
                for j in range(len(theta)):
                    term[i, j] = 0.5 * np.trace(ai @ (inv_cov @ d_covs[j]))
            fim = fim + term
            meta["CovTermIncluded"] = True

        return fim, meta

    # ------------------------------------------------------------------
    # スペクトル診断
    # ------------------------------------------------------------------
    @staticmethod
    def _spectral_metrics(eigenvalues: np.ndarray, rank_tol: float) -> Dict[str, Any]:
        ev = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
        ev = np.sort(ev)[::-1]
        max_ev = ev[0] if len(ev) else 0.0
        min_ev = ev[-1] if len(ev) else 0.0
        eps_guard = 1e-300
        cond = (max_ev + eps_guard) / (min_ev + eps_guard) if max_ev > 0 else 1.0
        numerical_rank = int(np.sum(ev > max_ev * rank_tol)) if max_ev > 0 else 0
        total = np.sum(ev)
        if total > 0:
            p = ev / total
            p = p[p > 0]
            eff_rank = float(np.exp(-np.sum(p * np.log(p))))
        else:
            eff_rank = 0.0
        return {
            "Eigenvalues": ev,
            "ConditionNumber": float(cond),
            "Log10ConditionNumber": float(np.log10(cond)) if cond > 0 else 0.0,
            "EffectiveRank": eff_rank,
            "NumericalRank": numerical_rank,
            "MaxEigenvalue": float(max_ev),
            "MinEigenvalue": float(min_ev),
        }

    def evaluate(
        self,
        theta: np.ndarray,
        observed_data: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
        cov_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        ridge_rel: float = 1e-10,
        cond_threshold: float = 1e12,
        include_cov_term: bool = False,
        rank_tol: float = 1e-10,
    ) -> Dict[str, Any]:
        """
        生 FIM と相関化 FIM の双方についてスペクトル診断を返す。

        相関化 FIM: Fc_ij = F_ij / sqrt(F_ii * F_jj)。
        パラメータの対角スケーリング theta_j -> c_j * theta_j に対して
        その固有値スペクトルは不変であり、単位の取り方に依存しない
        「構造的な」硬直度・実効ランクを与える。
        """
        fim, meta = self.compute_fisher_information_matrix(
            theta,
            observed_data=observed_data,
            covariance=covariance,
            cov_func=cov_func,
            ridge_rel=ridge_rel,
            cond_threshold=cond_threshold,
            include_cov_term=include_cov_term,
        )
        fim = 0.5 * (fim + fim.T)
        raw = self._spectral_metrics(np.linalg.eigvalsh(fim), rank_tol)

        d = np.diag(fim).copy()
        result: Dict[str, Any] = {"FIM": fim, **meta, "Raw": raw}
        if np.all(d > 0):
            inv_sqrt = 1.0 / np.sqrt(d)
            fim_corr = fim * np.outer(inv_sqrt, inv_sqrt)
            fim_corr = 0.5 * (fim_corr + fim_corr.T)
            result["FIM_correlation"] = fim_corr
            result["Scaled"] = self._spectral_metrics(
                np.linalg.eigvalsh(fim_corr), rank_tol
            )
        else:
            result["FIM_correlation"] = None
            result["Scaled"] = None
            warnings.warn(
                "FIM の対角成分に 0 以下のものがあるため相関化 FIM を作れませんでした"
                "（そのパラメータ方向は観測に全く効いていません）。",
                RuntimeWarning,
            )
        return result
