"""
汎用・情報幾何学的診断フレームワーク v3
UniversalInformationGeometricDiagnosticsV3

v2 の検証済みコア（数値微分・FIM 組み立て・Raw/Scaled スペクトル）は
一切変更せず、サブクラスとして以下を「追加」する。いずれも 2026-11-09 の
ボイド/重力波検証で「v2 が自分では言えなかったこと」への対応である。

  1. 絶対スケールの併記 (Absolute)
     - r_eff や条件数はスケールフリーであり、「効果の絶対量が桁で足りない」
       事態に沈黙する。各固有方向について、方向 SNR = sqrt(lambda)
       （その方向へ単位量ずらしたときの Δχ² の平方根）、方向 σ = 1/sqrt(lambda)
       （クラメール・ラオ下限）、および目標 SNR に届くために必要な
       情報量の倍率 (target_snr^2 / lambda) を返す。FIM が観測の反復回数に
       比例する系では、この倍率は「同じ観測を何倍に増やせば届くか」に等しい。
  2. 邪魔者（nuisance）の周辺化
     - nuisance_basis: 観測空間の線形基底 B（例: 検出器ごとの定数振幅）。
       線形振幅をプロファイル（Schur 補完）した FIM を返す。フィットで
       吸収されてしまう成分は、診断の段階で正しくゼロとして現れる。
     - nuisance_params: パラメータ空間の Schur 補完による周辺化。
  3. 貪欲選定の内訳 (greedy_select)
     - 各ステップで、勝者と次点について「勾配の大きさ ‖J_i‖²」と
       「ノイズ 1/σ_i²」の寄与を分けて返し、勝因ラベル (driver) と
       マージンを付ける。注目点が信号で勝ったのかノイズ構造で勝ったのかを、
       掘り直さずにその場で言えるようにする。
"""

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from universal_info_geo_diagnostics_v2 import (
    UniversalInformationGeometricDiagnosticsV2,
)


class UniversalInformationGeometricDiagnosticsV3(
    UniversalInformationGeometricDiagnosticsV2
):
    # ------------------------------------------------------------------
    # 1. 絶対スケール
    # ------------------------------------------------------------------
    @staticmethod
    def absolute_scale(
        eigenvalues: np.ndarray, target_snr: float = 1.0
    ) -> Dict[str, Any]:
        """固有値列に対する絶対スケール指標。

        DirectionSNR[k] = sqrt(lambda_k): 固有ベクトル方向への単位変位が
        生む Δχ² の平方根。その方向の「信号対雑音」の絶対量である。
        InfoMultiplier[k] = target_snr^2 / lambda_k: 同じ観測を何倍に
        増やせば（FIM が何倍になれば）その方向が target_snr に届くか。
        """
        ev = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
        ev = np.sort(ev)[::-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.sqrt(ev)
            sigma = np.where(ev > 0, 1.0 / snr, np.inf)
            mult = np.where(ev > 0, target_snr ** 2 / ev, np.inf)
        return {
            "TargetSNR": float(target_snr),
            "DirectionSNR": snr,
            "DirectionSigmaCR": sigma,
            "InfoMultiplierForTargetSNR": mult,
            "StiffestDirectionSNR": float(snr[0]) if len(snr) else 0.0,
            "SloppiestDirectionSNR": float(snr[-1]) if len(snr) else 0.0,
            "DirectionsAtOrAboveTarget": int(np.sum(snr >= target_snr)),
        }

    # ------------------------------------------------------------------
    # 2. 邪魔者の周辺化
    # ------------------------------------------------------------------
    def profiled_fim(
        self,
        theta: np.ndarray,
        nuisance_basis: np.ndarray,
        observed_data: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
        cov_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        ridge_rel: float = 1e-10,
        cond_threshold: float = 1e12,
    ):
        """観測空間の線形 nuisance 基底 B をプロファイルした FIM。

        F_prof = JᵀΣ⁻¹J − (JᵀΣ⁻¹B)(BᵀΣ⁻¹B)⁺(BᵀΣ⁻¹J)
        線形振幅を最尤で落とした後の情報であり、B の張る方向にのみ
        応答するパラメータ方向は厳密にゼロになる（吸収の再現）。
        """
        theta = np.asarray(theta, dtype=float)
        B = np.asarray(nuisance_basis, dtype=float)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        f0 = self.model_func(theta)
        J = self.compute_jacobian(theta, base_output=f0)
        cov = self._resolve_covariance(theta, observed_data, covariance, cov_func)
        solved_j, path, cond = self._stabilized_solve(
            cov, J, ridge_rel, cond_threshold)
        solved_b, _, _ = self._stabilized_solve(cov, B, ridge_rel, cond_threshold)
        f_full = J.T @ solved_j
        cross = J.T @ solved_b            # Jᵀ Σ⁻¹ B  (N x k)
        sbb = B.T @ solved_b              # Bᵀ Σ⁻¹ B  (k x k)
        f_prof = f_full - cross @ np.linalg.pinv(sbb, rcond=1e-12) @ cross.T
        meta = {"InversePath": path, "CovConditionNumber": cond,
                "NuisanceBasisDim": int(B.shape[1])}
        return 0.5 * (f_prof + f_prof.T), meta

    @staticmethod
    def marginalize_fim(F: np.ndarray, nuisance_idx: Sequence[int]) -> np.ndarray:
        """パラメータ空間の Schur 補完: F_eff = F_ss − F_sn F_nn⁺ F_ns。"""
        F = np.asarray(F, dtype=float)
        n = F.shape[0]
        keep = [i for i in range(n) if i not in set(nuisance_idx)]
        nuis = list(nuisance_idx)
        if not nuis:
            return F.copy()
        Fss = F[np.ix_(keep, keep)]
        Fsn = F[np.ix_(keep, nuis)]
        Fnn = F[np.ix_(nuis, nuis)]
        out = Fss - Fsn @ np.linalg.pinv(Fnn, rcond=1e-12) @ Fsn.T
        return 0.5 * (out + out.T)

    def _metrics_block(self, fim: np.ndarray, rank_tol: float,
                       target_snr: float) -> Dict[str, Any]:
        fim = 0.5 * (fim + fim.T)
        raw = self._spectral_metrics(np.linalg.eigvalsh(fim), rank_tol)
        block: Dict[str, Any] = {"FIM": fim, "Raw": raw,
                                 "AbsoluteRaw": self.absolute_scale(
                                     raw["Eigenvalues"], target_snr)}
        d = np.diag(fim)
        if np.all(d > 0):
            inv_sqrt = 1.0 / np.sqrt(d)
            fc = fim * np.outer(inv_sqrt, inv_sqrt)
            fc = 0.5 * (fc + fc.T)
            scaled = self._spectral_metrics(np.linalg.eigvalsh(fc), rank_tol)
            block["Scaled"] = scaled
            block["AbsoluteScaled"] = self.absolute_scale(
                scaled["Eigenvalues"], target_snr)
        else:
            block["Scaled"] = None
            block["AbsoluteScaled"] = None
        return block

    # ------------------------------------------------------------------
    # evaluate（v2 の出力に Absolute / 周辺化ブロックを追加）
    # ------------------------------------------------------------------
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
        target_snr: float = 1.0,
        nuisance_basis: Optional[np.ndarray] = None,
        nuisance_params: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        result = super().evaluate(
            theta,
            observed_data=observed_data,
            covariance=covariance,
            cov_func=cov_func,
            ridge_rel=ridge_rel,
            cond_threshold=cond_threshold,
            include_cov_term=include_cov_term,
            rank_tol=rank_tol,
        )
        result["Absolute"] = {
            "Raw": self.absolute_scale(result["Raw"]["Eigenvalues"], target_snr),
            "Scaled": (self.absolute_scale(result["Scaled"]["Eigenvalues"],
                                           target_snr)
                       if result.get("Scaled") is not None else None),
        }
        base_fim = result["FIM"]
        if nuisance_basis is not None:
            f_prof, meta = self.profiled_fim(
                theta, nuisance_basis, observed_data=observed_data,
                covariance=covariance, cov_func=cov_func,
                ridge_rel=ridge_rel, cond_threshold=cond_threshold)
            block = self._metrics_block(f_prof, rank_tol, target_snr)
            block.update(meta)
            result["NuisanceProfiled"] = block
            base_fim = f_prof
        if nuisance_params is not None:
            f_marg = self.marginalize_fim(base_fim, list(nuisance_params))
            result["Marginalized"] = self._metrics_block(
                f_marg, rank_tol, target_snr)
            result["Marginalized"]["KeptParams"] = [
                i for i in range(base_fim.shape[0])
                if i not in set(nuisance_params)]
        return result

    # ------------------------------------------------------------------
    # 3. 貪欲選定（内訳つき）
    # ------------------------------------------------------------------
    def greedy_select(
        self,
        theta: np.ndarray,
        covariance: np.ndarray,
        max_points: Optional[int] = None,
        ridge_rel: float = 1e-10,
    ) -> List[Dict[str, Any]]:
        """観測点の貪欲選定。各ステップに勝因の内訳とマージンを付ける。

        利得は logdet(F_S) の増分（F_S = J_Sᵀ Σ_SS⁻¹ J_S + 相対リッジ）。
        内訳は対角近似の分解: whitened score ≒ ‖J_i‖²/σ_i² を
        勾配部 grad_sq = ‖J_i‖² と ノイズ部 1/σ_i² に分けて勝者/次点で比べ、
        driver ∈ {gradient, noise, structure} を付ける
        （structure = 対角近似では説明できない相関・冗長性の効果で勝った場合）。
        """
        theta = np.asarray(theta, dtype=float)
        cov = np.asarray(covariance, dtype=float)
        J = self.compute_jacobian(theta)
        m = J.shape[0]
        sigma_i = np.sqrt(np.maximum(np.diag(cov), 0.0))
        grad_sq = np.sum(J ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            diag_score = np.where(sigma_i > 0, grad_sq / sigma_i ** 2, np.inf)

        def fim_of(sel: List[int]) -> np.ndarray:
            if not sel:
                return np.zeros((J.shape[1], J.shape[1]))
            Js = J[sel]
            Ss = cov[np.ix_(sel, sel)]
            solved, _, _ = self._stabilized_solve(Ss, Js, ridge_rel, 1e12)
            return Js.T @ solved

        n_par = J.shape[1]
        f_full = fim_of(list(range(m)))
        eps = ridge_rel * max(np.trace(f_full) / n_par, 1e-300)

        def logdet_f(sel: List[int]) -> float:
            F = fim_of(sel) + eps * np.eye(n_par)
            sign, ld = np.linalg.slogdet(F)
            return float(ld) if sign > 0 else -np.inf

        selected: List[int] = []
        remaining = list(range(m))
        base_ld = logdet_f([])
        records: List[Dict[str, Any]] = []
        n_pick = m if max_points is None else min(max_points, m)
        for _ in range(n_pick):
            gains = {i: logdet_f(selected + [i]) - base_ld for i in remaining}
            order = sorted(gains, key=gains.get, reverse=True)
            win = order[0]
            run = order[1] if len(order) > 1 else None
            gain_w = gains[win]
            gain_r = gains[run] if run is not None else 0.0
            # 内訳（対角近似）: 勝者 vs 次点
            log_grad = float(np.log(grad_sq[win] / grad_sq[run])) \
                if run is not None and grad_sq[run] > 0 else 0.0
            log_noise = float(2.0 * np.log(sigma_i[run] / sigma_i[win])) \
                if run is not None and sigma_i[win] > 0 else 0.0
            if run is None:
                driver = "structure"
            elif max(log_grad, log_noise) <= 0:
                driver = "structure"
            else:
                driver = "gradient" if log_grad >= log_noise else "noise"
            records.append({
                "Step": len(records) + 1,
                "Index": int(win),
                "LogdetGain": float(gain_w),
                "RunnerUpIndex": (int(run) if run is not None else None),
                "RunnerUpGain": float(gain_r),
                "GainMargin": float(gain_w / gain_r) if gain_r > 0 else np.inf,
                "DiagScore": float(diag_score[win]),
                "DiagScoreMargin": (float(diag_score[win] / diag_score[run])
                                    if run is not None and diag_score[run] > 0
                                    else np.inf),
                "GradSq": float(grad_sq[win]),
                "Sigma": float(sigma_i[win]),
                "LogGradContribution": log_grad,
                "LogNoiseContribution": log_noise,
                "Driver": driver,
                "CumLogdet": float(base_ld + gain_w),
            })
            selected.append(win)
            remaining.remove(win)
            base_ld += gain_w
        return records
