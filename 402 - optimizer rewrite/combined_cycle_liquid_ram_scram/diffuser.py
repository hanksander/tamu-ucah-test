'''
So the current model is: cowl shock → instantaneous normal shock → subsonic exit,
  with Pt ratio and exit Mach fixed purely by freestream.

  Physics-backed diffuser proposal (no code written)

  1. Geometry. Add a subsonic diffuser between the throat (station 4) and the combustor
   face (station 4.5). Parametrize by A_diff_exit / A_throat (= combustor inlet area /
  throat area) and a length or half-angle. Store A(x) for x_throat ≤ x ≤ x_combustor.
  This is a new block in 402inlet2.py, called from design_2ramp_shock_matched_inlet
  after place_throat_on_cowl_and_cowl_shock and returned in result alongside the
  existing geometry dicts.

  2. Shock position from back pressure. Replace the "immediate normal" assumption with
  solve_terminal_shock_position(result, p_back, Pt_after_cowl, Tt0):
  - Along the supersonic branch downstream of the throat, M_sup(x) from isentropic
  A(x)/A* = f(M).
  - At trial x_s: apply normal_shock(M_sup(x_s)) → M_sub(x_s), pt_ratio, p2/p1.
  - Isentropically diffuse subsonic from M_sub(x_s) at A(x_s) to A_exit → predicted
  p_exit.
  - Root-find x_s so p_exit == p_back. Bounds: p_back too low ⇒ shock swallowed
  (supersonic combustor entry / unstart toward scram); p_back too high ⇒ shock expelled
   past the throat ⇒ unstart.
  - Returns x_s, M_exit, pt_frac_after_terminal_shock.

  3. Off-design. evaluate_fixed_geometry_at_condition gains a p_back argument and calls
   the same routine. The value replacing pt_frac_after_immediate_normal_shock becomes a
   function of (M0, alt, alpha, p_back).

  4. pyCycle coupling in pyc_run.py. This is the critical structural change: inlet pt
  recovery now depends on combustor inlet static pressure, which depends on inlet Pt —
  an implicit loop.
  - Split compute_inlet_conditions so that the frozen part (everything up to post-cowl
  state: Pt_after_cowl, Tt0) is computed once, and the terminal-shock part is an
  OpenMDAO ExplicitComponent (e.g. DiffuserTerminalShock) exposing inputs Ps_back
  (wired from the combustor inlet static pressure, comb.Fl_I:stat:P) and outputs
  ram_recovery, MN_exit.
  - Feed those outputs into the existing pyc.Inlet as ram_recovery and into
  Fl_I:stat:MN at the combustor. Add the component to the Newton group so the solver
  closes Ps_back ↔ ram_recovery consistently. The SCRAM branch is untouched (shock is
  swallowed by assumption).

  5. Plots in plots_pycycle.py.
  - fig01_inlet_design_detail: extend the geometry drawing to render the diffuser walls
   from throat_upper_xy/throat_lower_xy out to the combustor face, and mark the solved
  shock location x_s as a vertical segment inside the diffuser.
  - fig02 grid and fig_inlet_pt_vs_mach: evaluate_fixed_geometry_at_condition calls now
   need a p_back — either use the converged combustor Ps from the cycle solve at each
  grid point, or sweep p_back/Pt0 as a second axis.
  - Add a new figure: shock position x_s and pt_recovery vs p_back/Pt_after_cowl at the
   design Mach, showing the unstart boundaries on both ends.

  Key risk. Coupling the inlet recovery to combustor Ps can destabilize the Newton
  solve near unstart. Expect to need bracketing / bounds on x_s inside
  solve_terminal_shock_position and a graceful failure mode (return the swallowed-shock
   result with a warning flag) when p_back is outside [p_swallow, p_expel].


# ============================================================================
# Replacement code for 402inlet2.py — points 1 and 2.
# Left commented for now. Paste into 402inlet2.py (it uses GAMMA, R,
# area_mach_ratio, normal_shock, invert_area_mach_ratio_supersonic, which
# already live there).
# ============================================================================
#
# # ----------------------------------------------------------------------------
# # Point 1: Subsonic diffuser geometry (station 4 -> station 4.5)
# # ----------------------------------------------------------------------------
# def build_subsonic_diffuser(T_upper, T_lower, h_throat, width_m,
#                             area_ratio_exit_to_throat,
#                             half_angle_deg=None, length_m=None,
#                             n_stations=51):
#     """
#     Build a straight-walled symmetric subsonic diffuser downstream of the
#     throat. Exactly one of (half_angle_deg, length_m) must be provided.
#
#     Area law: linear A(x) between A_throat and A_exit along the axial
#     coordinate. (Linear-area is the simplest consistent choice; swap in a
#     conical or bell law later without changing the shock solver.)
#
#     Returns a dict with axial stations, per-station area, upper/lower wall
#     points in the inlet (x,y) frame, and the combustor-face endpoints.
#     """
#     import numpy as np, math
#
#     if area_ratio_exit_to_throat <= 1.0:
#         raise ValueError("Diffuser exit/throat area ratio must exceed 1.")
#     if (half_angle_deg is None) == (length_m is None):
#         raise ValueError("Specify exactly one of half_angle_deg or length_m.")
#
#     A_throat = h_throat * width_m
#     A_exit   = area_ratio_exit_to_throat * A_throat
#     h_exit   = A_exit / width_m
#
#     if length_m is None:
#         # Each wall opens outward by (h_exit - h_throat)/2 over length L.
#         dh_per_side = 0.5 * (h_exit - h_throat)
#         length_m = dh_per_side / math.tan(math.radians(half_angle_deg))
#
#     x0 = 0.5 * (T_upper[0] + T_lower[0])          # throat axial station
#     y_mid = 0.5 * (T_upper[1] + T_lower[1])       # diffuser centerline y
#     x_exit = x0 + length_m
#
#     xs = np.linspace(x0, x_exit, n_stations)
#     s  = (xs - x0) / length_m                     # 0..1
#     A_of_x = A_throat + s * (A_exit - A_throat)
#     h_of_x = A_of_x / width_m
#
#     upper_wall = np.column_stack([xs, y_mid + 0.5 * h_of_x])
#     lower_wall = np.column_stack([xs, y_mid - 0.5 * h_of_x])
#
#     exit_upper = upper_wall[-1]
#     exit_lower = lower_wall[-1]
#
#     return {
#         "x_throat":      x0,
#         "x_exit":        x_exit,
#         "length_m":      float(length_m),
#         "A_throat":      float(A_throat),
#         "A_exit":        float(A_exit),
#         "h_throat":      float(h_throat),
#         "h_exit":        float(h_exit),
#         "area_ratio":    float(area_ratio_exit_to_throat),
#         "width_m":       float(width_m),
#         "y_centerline":  float(y_mid),
#         "x_stations":    xs,
#         "A_stations":    A_of_x,
#         "h_stations":    h_of_x,
#         "upper_wall_xy": upper_wall,
#         "lower_wall_xy": lower_wall,
#         "exit_upper_xy": exit_upper,
#         "exit_lower_xy": exit_lower,
#     }
#
#
# # Inserted near the end of design_2ramp_shock_matched_inlet, after
# # place_throat_on_cowl_and_cowl_shock(...) produces T_upper/T_lower:
# #
# #     diffuser = build_subsonic_diffuser(
# #         T_upper=throat_geom["T_upper"],
# #         T_lower=throat_geom["T_lower"],
# #         h_throat=throat["h_throat"],
# #         width_m=width_m,
# #         area_ratio_exit_to_throat=DIFFUSER_AREA_RATIO,   # from pyc_config
# #         half_angle_deg=DIFFUSER_HALF_ANGLE_DEG,          # from pyc_config
# #     )
# #     result["diffuser"] = diffuser
# #     result["combustor_face_xy_upper"] = diffuser["exit_upper_xy"]
# #     result["combustor_face_xy_lower"] = diffuser["exit_lower_xy"]
# #     result["A_combustor_face"] = diffuser["A_exit"]
#
#
# # ----------------------------------------------------------------------------
# # Point 2: Terminal normal-shock position driven by back pressure
# # ----------------------------------------------------------------------------
# def _static_over_total(M, gamma=GAMMA):
#     return (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (-gamma / (gamma - 1.0))
#
#
# def _subsonic_mach_from_area_ratio(A_over_Astar, tol=1e-10, max_iter=200):
#     """Subsonic branch of the isentropic area-Mach relation."""
#     if A_over_Astar < 1.0:
#         raise ValueError("A/A* must be >= 1 for a real subsonic solution.")
#     lo, hi = 1e-6, 1.0 - 1e-8
#     for _ in range(max_iter):
#         mid = 0.5 * (lo + hi)
#         f = area_mach_ratio(mid) - A_over_Astar
#         if abs(f) < tol:
#             return mid
#         # area_mach_ratio is monotonically decreasing on (0,1).
#         if f > 0.0:
#             lo = mid
#         else:
#             hi = mid
#     return 0.5 * (lo + hi)
#
#
# def _exit_static_pressure_for_shock_at(x_s, diffuser, Pt_after_cowl,
#                                        M_throat=1.0):
#     """
#     Given a trial shock axial station x_s in the diverging (supersonic)
#     region, propagate to the diffuser exit and return predicted exit static
#     pressure plus intermediate state (for diagnostics / final return).
#
#     Assumes isentropic supersonic expansion from M_throat=1 at A_throat to
#     M_sup(x_s) at A(x_s), a normal shock at x_s, then isentropic subsonic
#     diffusion to A_exit. Pt_after_cowl is the total pressure feeding the
#     throat (cowl-shock exit value, already includes all oblique losses).
#     """
#     import numpy as np
#
#     A_throat = diffuser["A_throat"]
#     A_exit   = diffuser["A_exit"]
#     xs       = diffuser["x_stations"]
#     A_stats  = diffuser["A_stations"]
#
#     if x_s <= xs[0] or x_s >= xs[-1]:
#         raise ValueError("Trial shock station outside diffuser.")
#
#     A_s = float(np.interp(x_s, xs, A_stats))
#
#     # Supersonic branch upstream of the shock (A* = A_throat since M=1 there).
#     M_sup = invert_area_mach_ratio_supersonic(A_s / A_throat)
#
#     # Normal shock at the trial station.
#     M_sub, p2_p1, _, _, pt2_pt1 = normal_shock(M_sup)
#     Pt_after_shock = Pt_after_cowl * pt2_pt1
#
#     # Downstream A* after the shock (subsonic branch). A* shrinks by pt ratio
#     # for choked-reference, but here we just use the subsonic area-Mach with
#     # the new A*_sub = A_s / area_mach_ratio(M_sub).
#     Astar_sub = A_s / area_mach_ratio(M_sub)
#     M_exit    = _subsonic_mach_from_area_ratio(A_exit / Astar_sub)
#
#     Ps_exit = Pt_after_shock * _static_over_total(M_exit)
#     return {
#         "x_s":            x_s,
#         "A_s":            A_s,
#         "M_sup":          M_sup,
#         "M_sub":          M_sub,
#         "pt_ratio_shock": pt2_pt1,
#         "Pt_after_shock": Pt_after_shock,
#         "M_exit":         M_exit,
#         "Ps_exit":        Ps_exit,
#     }
#
#
# def solve_terminal_shock_position(result, p_back, Pt_after_cowl, Tt0,
#                                   tol=1e-4, max_iter=80):
#     """
#     Find axial shock station x_s in the subsonic diffuser such that the
#     predicted exit static pressure equals p_back.
#
#     Returns dict with x_s, M_sup, M_sub, M_exit, Pt ratios, plus a status
#     flag describing swallow/expel outcomes.
#         status in {"normal", "swallowed", "expelled"}.
#     """
#     diffuser = result["diffuser"]
#     xs = diffuser["x_stations"]
#
#     # Bracket: shock just past the throat (weakest shock at M~1+) gives the
#     # HIGHEST exit Ps; shock near the exit (strongest shock at M_sup_max)
#     # gives the LOWEST exit Ps.
#     eps = 1e-4 * (xs[-1] - xs[0])
#     x_lo = xs[0] + eps        # weak-shock limit (high Ps_exit)
#     x_hi = xs[-1] - eps       # strong-shock limit (low  Ps_exit)
#
#     hi_state = _exit_static_pressure_for_shock_at(x_hi, diffuser, Pt_after_cowl)
#     lo_state = _exit_static_pressure_for_shock_at(x_lo, diffuser, Pt_after_cowl)
#
#     Ps_max = lo_state["Ps_exit"]   # highest achievable Ps_exit
#     Ps_min = hi_state["Ps_exit"]   # lowest  achievable Ps_exit
#
#     if p_back > Ps_max:
#         # Back pressure too high: shock expelled upstream of throat -> unstart.
#         return {"status": "expelled", "p_back": p_back,
#                 "Ps_max": Ps_max, "Ps_min": Ps_min,
#                 "pt_frac_after_terminal_shock":
#                     Pt_after_cowl / Pt_after_cowl,  # placeholder; caller
#                                                      # should treat as failure
#                 **lo_state}
#     if p_back < Ps_min:
#         # Back pressure too low: shock swallowed out of diffuser.
#         return {"status": "swallowed", "p_back": p_back,
#                 "Ps_max": Ps_max, "Ps_min": Ps_min,
#                 "pt_frac_after_terminal_shock":
#                     hi_state["Pt_after_shock"] / Pt_after_cowl,
#                 **hi_state}
#
#     # Bisection on x_s. Ps_exit(x_s) is monotone decreasing in x_s (stronger
#     # shock further downstream), so root exists and is unique.
#     lo, hi = x_lo, x_hi
#     state = None
#     for _ in range(max_iter):
#         mid = 0.5 * (lo + hi)
#         state = _exit_static_pressure_for_shock_at(mid, diffuser, Pt_after_cowl)
#         err = state["Ps_exit"] - p_back
#         if abs(err) < tol * max(p_back, 1.0):
#             break
#         if err > 0.0:       # exit Ps too high -> push shock downstream
#             lo = mid
#         else:
#             hi = mid
#
#     pt_frac_terminal = state["Pt_after_shock"] / Pt_after_cowl
#
#     # Exit static temperature and velocity (handy for the combustor face).
#     import math
#     T_exit = Tt0 / (1.0 + 0.5 * (GAMMA - 1.0) * state["M_exit"] ** 2)
#     a_exit = math.sqrt(GAMMA * R * T_exit)
#     V_exit = state["M_exit"] * a_exit
#
#     return {
#         "status":                        "normal",
#         "p_back":                        p_back,
#         "Ps_max":                        Ps_max,
#         "Ps_min":                        Ps_min,
#         "x_s":                           state["x_s"],
#         "A_s":                           state["A_s"],
#         "M_sup":                         state["M_sup"],
#         "M_sub":                         state["M_sub"],
#         "M_exit":                        state["M_exit"],
#         "Ps_exit":                       state["Ps_exit"],
#         "Pt_after_terminal_shock":       state["Pt_after_shock"],
#         "pt_frac_after_terminal_shock":  pt_frac_terminal,
#         "T_exit":                        T_exit,
#         "a_exit":                        a_exit,
#         "V_exit":                        V_exit,
#     }
#
#
# # ----------------------------------------------------------------------------
# # Point 3: Off-design — evaluate_fixed_geometry_at_condition gains p_back
# # ----------------------------------------------------------------------------
# # This is a drop-in replacement for the existing
# # evaluate_fixed_geometry_at_condition in 402inlet2.py (around line 1141).
# # The oblique-shock chain is unchanged; only the terminal-shock block at the
# # bottom is swapped out to use the diffuser + back-pressure solver.
# #
# # Signature change: add `p_back` (Pa) as a required kwarg. Callers in
# # pyc_run.py (compute_inlet_conditions) and plots_pycycle.py must pass it;
# # during design-point evaluation pass p_back = result["Ps_design_exit"] (or
# # similar), and in the cycle solve pass the combustor inlet static pressure.
#
#
# def evaluate_fixed_geometry_at_condition(result, M0, altitude_m, alpha_deg,
#                                          p_back):
#     """
#     Re-evaluate the shock system for a fixed geometry at a new flight
#     condition AND back pressure. Geometry (including the subsonic diffuser
#     from build_subsonic_diffuser) is frozen inside `result`.
#
#     The terminal normal shock is no longer pinned to the cowl lip. Its axial
#     station is solved from p_back via solve_terminal_shock_position.
#     """
#     import math
#
#     T0, p0, rho0, a0 = std_atmosphere_1976(altitude_m)
#     V0 = M0 * a0
#
#     theta_fore = result["theta_fore_deg"]
#     theta1     = result["theta1_deg"]
#     theta2     = result["theta2_deg"]
#     theta_cowl = result["theta_cowl_deg"]
#
#     P_fore  = result["forebody_xy"]
#     P0_xy   = result["nose_xy"]
#     P1      = result["break2_xy"]
#     C       = result["cowl_lip_xy"]
#     F       = result["ramp2_normal_foot_xy"]
#     T_lower = result["throat_lower_xy"]
#     T_upper = result["throat_upper_xy"]
#     focus   = result["shock_focus_xy"]
#
#     # ---- Forebody shock
#     dtheta_fore_eff = theta_fore + alpha_deg
#     if dtheta_fore_eff <= 0.0:
#         return {"success": False,
#                 "reason": "Non-positive forebody effective turn",
#                 "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
#     shf = oblique_shock(M0, dtheta_fore_eff)
#     if shf is None:
#         return {"success": False, "reason": "Forebody shock unattached",
#                 "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
#     beta_fore_rel, M_fore, p_fore_ratio, pt_fore_ratio = shf
#     shock_fore_abs = beta_fore_rel
#
#     # ---- Ramp 1
#     dtheta1 = theta1 - theta_fore
#     if dtheta1 <= 0.0:
#         return {"success": False,
#                 "reason": "Invalid ramp 1 turn from frozen geometry",
#                 "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
#     sh1 = oblique_shock(M_fore, dtheta1)
#     if sh1 is None:
#         return {"success": False, "reason": "Ramp 1 shock unattached",
#                 "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
#     beta1_rel, M1, p21, pt21 = sh1
#     shock1_abs = theta_fore + beta1_rel
#
#     # ---- Ramp 2
#     dtheta2 = theta2 - theta1
#     if dtheta2 <= 0.0:
#         return {"success": False,
#                 "reason": "Invalid ramp 2 turn from frozen geometry",
#                 "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
#     sh2 = oblique_shock(M1, dtheta2)
#     if sh2 is None:
#         return {"success": False, "reason": "Ramp 2 shock unattached",
#                 "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
#     beta2_rel, M2, p32, pt32 = sh2
#     shock2_abs = theta1 + beta2_rel
#
#     # ---- Cowl shock
#     cowl_turn_mag = theta2 - theta_cowl
#     if cowl_turn_mag <= 0.0:
#         return {"success": False, "reason": "Non-positive cowl turn",
#                 "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
#     shc = oblique_shock(M2, cowl_turn_mag)
#     if shc is None:
#         return {"success": False, "reason": "Cowl shock unattached",
#                 "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
#     beta_cowl_rel, M3, p43, pt43 = shc
#     cowl_shock_abs = theta2 - beta_cowl_rel
#
#     pt_frac_after_forebody = pt_fore_ratio
#     pt_frac_after_shock1   = pt_fore_ratio * pt21
#     pt_frac_after_shock2   = pt_fore_ratio * pt21 * pt32
#     pt_frac_after_cowl     = pt_fore_ratio * pt21 * pt32 * pt43
#
#     Tt0 = total_temperature(T0, M0)
#     Pt0 = p0 * (1.0 + 0.5 * (GAMMA - 1.0) * M0 * M0) ** (GAMMA / (GAMMA - 1.0))
#     Pt_after_cowl = Pt0 * pt_frac_after_cowl
#
#     # ---- Terminal shock: driven by back pressure inside the diffuser.
#     if "diffuser" not in result:
#         return {"success": False,
#                 "reason": "Frozen geometry has no diffuser block. Rebuild "
#                           "with build_subsonic_diffuser first.",
#                 "M0": M0, "alpha_deg": alpha_deg, "p_back": p_back}
#
#     terminal = solve_terminal_shock_position(result, p_back,
#                                              Pt_after_cowl, Tt0)
#
#     # Propagate unstart outcomes up to the caller without crashing the sweep.
#     if terminal["status"] in ("expelled", "swallowed"):
#         return {
#             "success":        False,
#             "reason":         f"Terminal shock {terminal['status']} "
#                               f"(p_back={p_back:.1f} Pa, "
#                               f"Ps range=[{terminal['Ps_min']:.1f}, "
#                               f"{terminal['Ps_max']:.1f}] Pa).",
#             "status":         terminal["status"],
#             "M0":             M0,
#             "alpha_deg":      alpha_deg,
#             "p_back":         p_back,
#             "Pt_after_cowl":  Pt_after_cowl,
#             "terminal":       terminal,
#         }
#
#     pt_frac_after_terminal = (pt_frac_after_cowl
#                               * terminal["pt_frac_after_terminal_shock"])
#
#     return {
#         "success":        True,
#         "status":         terminal["status"],
#         "M0":             M0,
#         "alpha_deg":      alpha_deg,
#         "p_back":         p_back,
#         "V0_ms":          V0,
#
#         "theta_fore_deg":  theta_fore,
#         "theta1_deg":      theta1,
#         "theta2_deg":      theta2,
#         "theta_cowl_deg":  theta_cowl,
#
#         "shock_fore_abs_deg": shock_fore_abs,
#         "shock1_abs_deg":     shock1_abs,
#         "shock2_abs_deg":     shock2_abs,
#         "cowl_shock_abs_deg": cowl_shock_abs,
#
#         "M_after_forebody_shock":  M_fore,
#         "M_after_shock1":          M1,
#         "M_after_shock2":          M2,
#         "M_after_cowl_shock":      M3,
#
#         # Terminal-shock fields (replace the old *_immediate_normal_shock
#         # keys). Downstream consumers in pyc_run / plots_pycycle must be
#         # updated to read these names.
#         "x_terminal_shock":            terminal["x_s"],
#         "A_at_terminal_shock":         terminal["A_s"],
#         "M_before_terminal_shock":     terminal["M_sup"],
#         "M_after_terminal_shock":      terminal["M_sub"],
#         "M_at_combustor_face":         terminal["M_exit"],
#         "Ps_at_combustor_face":        terminal["Ps_exit"],
#         "V_at_combustor_face_ms":      terminal["V_exit"],
#
#         "pt_frac_after_forebody_shock":   pt_frac_after_forebody,
#         "pt_frac_after_shock1":           pt_frac_after_shock1,
#         "pt_frac_after_shock2":           pt_frac_after_shock2,
#         "pt_frac_after_cowl_shock":       pt_frac_after_cowl,
#         "pt_frac_after_terminal_shock":   pt_frac_after_terminal,
#
#         # Back-compat alias so existing call sites that read
#         # "pt_frac_after_immediate_normal_shock" keep working until migrated.
#         "pt_frac_after_immediate_normal_shock": pt_frac_after_terminal,
#         "M_after_immediate_normal_shock":       terminal["M_exit"],
#         "V_after_immediate_normal_shock_ms":    terminal["V_exit"],
#
#         "forebody_xy":         P_fore,
#         "nose_xy":             P0_xy,
#         "break2_xy":           P1,
#         "cowl_lip_xy":         C,
#         "ramp2_normal_foot_xy": F,
#         "throat_lower_xy":     T_lower,
#         "throat_upper_xy":     T_upper,
#         "shock_focus_xy":      focus,
#
#         "diffuser":            result["diffuser"],
#         "Pt_after_cowl":       Pt_after_cowl,
#         "Tt0":                 Tt0,
#     }
#
#
# # ----------------------------------------------------------------------------
# # Point 4: pyCycle coupling in pyc_run.py
# #
# # The inlet's terminal-shock Pt recovery now depends on combustor inlet
# # static pressure, which in turn depends on inlet Pt. That is an implicit
# # loop — it must be closed by a solver, not pre-computed.
# #
# # Strategy:
# #   (a) Split the frozen oblique-shock chain (M0, alt, alpha -> Pt_after_cowl,
# #       Tt0) out of compute_inlet_conditions. That piece stays explicit and
# #       runs once per design point.
# #   (b) Wrap solve_terminal_shock_position in an OpenMDAO ExplicitComponent
# #       (DiffuserTerminalShock) that takes Ps_back as an input and emits
# #       ram_recovery and MN_exit.
# #   (c) Wire comb.Fl_I:stat:P -> diff.Ps_back and diff.ram_recovery ->
# #       inlet.ram_recovery, diff.MN_exit -> burner.Fl_I:stat:MN. Put the
# #       group under a Newton solver so Ps_back and ram_recovery converge.
# #   (d) SCRAM branch is unchanged: shock is swallowed by assumption, so the
# #       new component is bypassed in SCRAM cycles.
# #
# # LINES FLAGGED BELOW with "### UNSTART" or "### NEWTON" are the spots most
# # likely to need bracketing / damping / relaxation when the solver pushes
# # Ps_back outside the stable operating window.
# # ----------------------------------------------------------------------------
#
#
# # ---- pyc_run.py replacement code -------------------------------------------
# # import openmdao.api as om
# # import numpy as np
# #
# # # _inlet2 and _get_inlet_design() are already defined in pyc_run.py.
# #
# #
# # def compute_precowl_state(M0, alt_m, alpha_deg=0.0):
# #     """
# #     Explicit, non-iterative: run the frozen oblique-shock chain up to the
# #     cowl-shock exit and return Pt_after_cowl, Tt0, M3, plus a success flag.
# #     Separated so it can be called once per design point and its outputs
# #     fed into the Newton-looped DiffuserTerminalShock component.
# #     """
# #     design = _get_inlet_design()
# #     # Use a sentinel p_back; we only need the pre-terminal fields. If the
# #     # new evaluate_fixed_geometry_at_condition short-circuits on p_back ==
# #     # None, even better — otherwise ignore terminal fields from this call.
# #     case = _inlet2.evaluate_fixed_geometry_at_condition(
# #         design, M0=M0, altitude_m=alt_m, alpha_deg=alpha_deg,
# #         p_back=1.0,  ### UNSTART: sentinel; terminal fields discarded below.
# #     )
# #     return design, case
# #
# #
# # class DiffuserTerminalShock(om.ExplicitComponent):
# #     """
# #     Solves the terminal normal shock position from combustor back pressure.
# #
# #     Inputs
# #     ------
# #     Ps_back : float   combustor inlet static pressure [Pa]
# #
# #     Outputs
# #     -------
# #     ram_recovery : float   Pt_exit / Pt0 including oblique chain + terminal
# #     MN_exit      : float   subsonic Mach at the diffuser exit
# #     Pt_after_cowl : float  diagnostic
# #     x_shock       : float  diagnostic (shock axial station)
# #     unstart_flag  : float  0 normal, +1 expelled, -1 swallowed
# #     """
# #
# #     def initialize(self):
# #         self.options.declare('design')         # frozen geometry dict
# #         self.options.declare('M0', types=float)
# #         self.options.declare('alt_m', types=float)
# #         self.options.declare('alpha_deg', default=0.0, types=float)
# #         self.options.declare('isolator_pt_recovery', default=1.0,
# #                              types=float)
# #
# #     def setup(self):
# #         self.add_input('Ps_back', val=5e4, units='Pa')
# #         self.add_output('ram_recovery', val=0.5)
# #         self.add_output('MN_exit', val=0.3)
# #         self.add_output('Pt_after_cowl_Pa', val=1e5, units='Pa')
# #         self.add_output('x_shock', val=0.0)
# #         self.add_output('unstart_flag', val=0.0)
# #         self.declare_partials('*', '*', method='fd')   ### NEWTON: FD is
# #         # cheap here but noisy near unstart boundaries where Ps_exit(x_s)
# #         # flattens. Switch to CS or analytic partials if Newton stalls.
# #
# #     def compute(self, inputs, outputs):
# #         design   = self.options['design']
# #         M0       = self.options['M0']
# #         alt_m    = self.options['alt_m']
# #         alpha    = self.options['alpha_deg']
# #         iso_pt   = self.options['isolator_pt_recovery']
# #         Ps_back  = float(inputs['Ps_back'][0])
# #
# #         # Re-run the frozen chain with this Ps_back.
# #         case = _inlet2.evaluate_fixed_geometry_at_condition(
# #             design, M0=M0, altitude_m=alt_m, alpha_deg=alpha,
# #             p_back=Ps_back,
# #         )
# #
# #         if not case.get('success', False):
# #             # Unstart handling. Do NOT raise — Newton will take wild steps
# #             # early on; we need to stay numerically finite so the solver
# #             # can back off.
# #             status = case.get('status', 'unknown')
# #             term   = case.get('terminal', {})
# #             Pt_ac  = case.get('Pt_after_cowl',
# #                               term.get('Pt_after_shock', 1.0))
# #             if status == 'expelled':
# #                 # Shock past throat: collapse to strongest-shock-at-throat
# #                 # estimate so the residual still pushes Ps_back down.
# #                 outputs['ram_recovery'] = 0.1 * iso_pt        ### UNSTART
# #                 outputs['MN_exit']      = 0.2                 ### UNSTART
# #                 outputs['unstart_flag'] = +1.0
# #             elif status == 'swallowed':
# #                 # Shock swept out: use weakest-shock-at-exit estimate.
# #                 outputs['ram_recovery'] = (
# #                     term.get('pt_frac_after_terminal_shock', 0.9)
# #                     * iso_pt)                                 ### UNSTART
# #                 outputs['MN_exit']      = 0.9                 ### UNSTART
# #                 outputs['unstart_flag'] = -1.0
# #             else:
# #                 outputs['ram_recovery'] = 0.05 * iso_pt       ### NEWTON
# #                 outputs['MN_exit']      = 0.3
# #                 outputs['unstart_flag'] = +1.0
# #             outputs['Pt_after_cowl_Pa'] = Pt_ac
# #             outputs['x_shock']          = 0.0
# #             return
# #
# #         # Normal (on-design) branch.
# #         pt_frac_total = case['pt_frac_after_terminal_shock'] * iso_pt
# #         MN_exit       = float(case['M_at_combustor_face'])
# #
# #         # Clip to keep pyCycle Inlet happy; the physical solution may dip
# #         # very low right at unstart, and Inlet balks at MN_exit > ~0.95.
# #         MN_exit = float(np.clip(MN_exit, 0.05, 0.95))          ### NEWTON
# #
# #         outputs['ram_recovery']     = float(pt_frac_total)
# #         outputs['MN_exit']          = MN_exit
# #         outputs['Pt_after_cowl_Pa'] = float(case['Pt_after_cowl'])
# #         outputs['x_shock']          = float(case['x_terminal_shock'])
# #         outputs['unstart_flag']     = 0.0
# #
# #
# # # ---- pyc_ram_cycle.py replacement ------------------------------------------
# # # In RamCycle.setup(), after the existing add_subsystem calls, insert:
# # #
# # #     self.add_subsystem(
# # #         'diff',
# # #         DiffuserTerminalShock(
# # #             design=_get_inlet_design(),
# # #             M0=self.options['design_M0'],
# # #             alt_m=self.options['design_alt_m'],
# # #             alpha_deg=self.options['design_alpha_deg'],
# # #             isolator_pt_recovery=ISOLATOR_PT_RECOVERY,
# # #         ),
# # #         promotes_inputs=[],
# # #     )
# # #
# # #     # Couple the loop.
# # #     self.connect('burner.Fl_I:stat:P', 'diff.Ps_back')
# # #     self.connect('diff.ram_recovery',  'inlet.ram_recovery')
# # #     self.connect('diff.MN_exit',       'burner.Fl_I:stat:MN')
# # #
# # #     # Newton to close Ps_back <-> ram_recovery.
# # #     newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
# # #     newton.options['maxiter']           = 40
# # #     newton.options['atol']              = 1e-6
# # #     newton.options['rtol']              = 1e-6
# # #     newton.options['iprint']            = 2
# # #     newton.options['err_on_non_converge'] = False   ### NEWTON: do not
# # #     # abort the whole sweep if one grid point lives at an unstart edge.
# # #
# # #     ls = newton.linesearch = om.ArmijoGoldsteinLS()
# # #     ls.options['maxiter']      = 10
# # #     ls.options['bound_enforcement'] = 'scalar'      ### NEWTON: prevents
# # #     # the solver from stepping Ps_back negative on the first iteration.
# # #     ls.options['print_bound_enforce'] = True
# # #
# # #     self.linear_solver = om.DirectSolver()
# # #
# # # SCRAM cycle (pyc_scram_cycle.py) is NOT modified — the shock is assumed
# # # swallowed past the throat so ram_recovery there is the cowl-shock value
# # # and MN_exit is supersonic, as today.
# #
# #
# # # ---- compute_inlet_conditions becomes two-stage ---------------------------
# # # def compute_inlet_conditions(M0, alt_m, mode, ramp_angles=None,
# # #                              alpha_deg=0.0):
# # #     """
# # #     RAM mode: returns (None, None) because ram_recovery and MN_exit are now
# # #     owned by DiffuserTerminalShock inside the Newton loop. Callers must
# # #     not push these into pyc.Inlet by hand; instead, wire the component in
# # #     RamCycle.setup() (see above).
# # #
# # #     SCRAM mode: unchanged — shock is swallowed.
# # #     """
# # #     del ramp_angles
# # #
# # #     if mode == 'ram':
# # #         return None, None     ### NEWTON: signal to caller that the
# # #         # RamCycle owns these values via DiffuserTerminalShock.
# # #
# # #     design = _get_inlet_design()
# # #     case = _inlet2.evaluate_fixed_geometry_at_condition(
# # #         design, M0=M0, altitude_m=alt_m, alpha_deg=alpha_deg,
# # #         p_back=1.0,   ### UNSTART: SCRAM ignores terminal fields.
# # #     )
# # #     if not case.get('success', False):
# # #         Pt_ratio = pi_milspec(M0) * ISOLATOR_PT_RECOVERY
# # #         exit_MN  = max(M0 * 0.75, 1.05)
# # #         return float(Pt_ratio), float(exit_MN)
# # #
# # #     Pt_ratio = case['pt_frac_after_cowl_shock'] * ISOLATOR_PT_RECOVERY
# # #     exit_MN  = float(case['M_after_cowl_shock'])
# # #     return float(Pt_ratio), float(exit_MN)
#
#
# # ============================================================================
# # Point 5 (partial): plots_pycycle.py changes + combustor dimension coupling
# #
# # The subsonic diffuser sits between the throat and the combustor face. That
# # means two things for plotting / sizing:
# #
# #   A. The combustor inlet area is NO LONGER the throat area. It is the
# #      diffuser exit area, A_exit = area_ratio * A_throat. The combustor
# #      height (and therefore its length, since volume = L* * A_throat_nozzle
# #      is unchanged) must be rebuilt from the diffuser exit, not from
# #      throat_upper_xy / throat_lower_xy.
# #
# #   B. The flowpath drawing must add diffuser walls between the throat
# #      station and the combustor face, and axial station x3 ("Inlet exit")
# #      should shift from duct_x0 (old throat) to duct_x0 + diffuser length.
# #      A new annotation marks the terminal shock x_s inside the diffuser.
# # ============================================================================
#
#
# # ---- pyc_run.compute_combustor_geometry update -----------------------------
# # Accept the new diffuser block and derive height_m from the diffuser exit
# # rather than the throat. Keep the old fallback so SCRAM / legacy callers
# # that never built a diffuser still work.
# #
# # def compute_combustor_geometry(nozzle_throat_area, combustor_L_star,
# #                                design=None, width_m=None, height_m=None):
# #     if nozzle_throat_area <= 0.0:
# #         raise ValueError("nozzle_throat_area must be positive.")
# #     if combustor_L_star <= 0.0:
# #         raise ValueError("combustor_L_star must be positive.")
# #
# #     if width_m is None:
# #         width_m = INLET_DESIGN_WIDTH_M
# #
# #     if height_m is None:
# #         if design is None:
# #             raise ValueError("height_m or design must be provided.")
# #         # NEW: prefer the diffuser exit height when a diffuser is present.
# #         diff = design.get("diffuser") if isinstance(design, dict) else None
# #         if diff is not None:
# #             height_m = float(diff["h_exit"])          # <-- combustor face
# #         elif "throat_height_m" in design:
# #             height_m = float(design["throat_height_m"])
# #         else:
# #             t_up = np.asarray(design["throat_upper_xy"], dtype=float)
# #             t_lo = np.asarray(design["throat_lower_xy"], dtype=float)
# #             height_m = float(abs(t_up[1] - t_lo[1]))
# #
# #     if width_m <= 0.0 or height_m <= 0.0:
# #         raise ValueError("Combustor width and height must be positive.")
# #
# #     combustor_area   = width_m * height_m
# #     combustor_volume = combustor_L_star * nozzle_throat_area
# #     combustor_length = combustor_volume / combustor_area
# #     # Note: because combustor_area grew by area_ratio, combustor_length
# #     # SHRINKS by the same factor at fixed L* and nozzle throat area. That
# #     # is physical: a wider burner needs less axial length for the same
# #     # residence volume. Watch for L/D < ~1 at high area ratios — if that
# #     # happens, revisit L* or cap the diffuser area ratio.
# #
# #     return {
# #         "L_star":                float(combustor_L_star),
# #         "width_m":               float(width_m),
# #         "height_m":              float(height_m),
# #         "cross_section_area_m2": float(combustor_area),
# #         "throat_area_m2":        float(nozzle_throat_area),
# #         "volume_m3":             float(combustor_volume),
# #         "length_m":              float(combustor_length),
# #     }
#
#
# # ---- plots_pycycle._flowpath_layout update ---------------------------------
# # Insert a diffuser block between the throat and the combustor face. The
# # combustor now starts at duct_x0 + diffuser.length_m, and its inlet area
# # matches the diffuser exit.
# #
# # def _flowpath_layout(design, design_cycle, combustor_L_star=...,
# #                      converging_length=..., diverging_length=...,
# #                      throat_angle_deg=..., exit_angle_deg=...,
# #                      n_points=...):
# #     fore = np.asarray(design['forebody_xy'], dtype=float)
# #     t_up = np.asarray(design['throat_upper_xy'], dtype=float)
# #     t_lo = np.asarray(design['throat_lower_xy'], dtype=float)
# #
# #     throat_h   = float(abs(t_up[1] - t_lo[1]))
# #     throat_x0  = max(t_up[0], t_lo[0])
# #     y_center   = 0.5 * (t_up[1] + t_lo[1])
# #
# #     # NEW: pull diffuser contour from frozen geometry.
# #     diff = design.get('diffuser')
# #     if diff is None:
# #         # Legacy path: no diffuser -> combustor sits directly at throat.
# #         diff_len = 0.0
# #         diff_h_exit = throat_h
# #         diff_upper = np.array([[throat_x0, y_center + 0.5 * throat_h]])
# #         diff_lower = np.array([[throat_x0, y_center - 0.5 * throat_h]])
# #     else:
# #         diff_len    = float(diff['length_m'])
# #         diff_h_exit = float(diff['h_exit'])
# #         diff_upper  = np.asarray(diff['upper_wall_xy'], dtype=float)
# #         diff_lower  = np.asarray(diff['lower_wall_xy'], dtype=float)
# #
# #     duct_x0 = throat_x0 + diff_len        # combustor entrance (station 3)
# #     duct_h  = diff_h_exit                 # combustor inlet height
# #     duct_y_lo = y_center - 0.5 * duct_h
# #
# #     A_throat = float(design_cycle['nozzle_throat_area'])
# #     combustor = design_cycle.get('combustor_geometry')
# #     if combustor is None or abs(float(combustor.get('L_star', np.nan))
# #                                 - combustor_L_star) > 1.0e-12:
# #         combustor = pyc_run.compute_combustor_geometry(
# #             nozzle_throat_area=A_throat,
# #             combustor_L_star=combustor_L_star,
# #             design=design,           # carries 'diffuser' now
# #         )
# #
# #     duct_len = float(combustor['length_m'])
# #     duct_x1  = duct_x0 + duct_len
# #
# #     A_exit  = float(design_cycle['nozzle_exit_area'])
# #     A_inlet = float(combustor['cross_section_area_m2'])
# #     bell = nozzle_design.generate_bell_contour(
# #         inlet_area=max(A_inlet, A_throat * 1.01),
# #         throat_area=A_throat,
# #         exit_area=A_exit,
# #         converging_length=converging_length,
# #         diverging_length=diverging_length,
# #         throat_angle_deg=throat_angle_deg,
# #         exit_angle_deg=exit_angle_deg,
# #         n_points=n_points,
# #     )
# #     x_shift = duct_x1 - bell['x'][0]
# #     bx = bell['x'] + x_shift
# #
# #     return {
# #         'fore':       fore,
# #         't_up':       t_up,
# #         't_lo':       t_lo,
# #         'throat_h':   throat_h,
# #         'throat_x0':  throat_x0,
# #         # Diffuser contour so figures can draw it.
# #         'diff_upper': diff_upper,
# #         'diff_lower': diff_lower,
# #         'diff_len':   diff_len,
# #         'diff_h_exit': diff_h_exit,
# #         # Combustor block — now starts at the diffuser exit.
# #         'duct_h':     duct_h,
# #         'duct_y_lo':  duct_y_lo,
# #         'duct_x0':    duct_x0,
# #         'duct_len':   duct_len,
# #         'duct_x1':    duct_x1,
# #         'combustor':  combustor,
# #         'A_inlet':    A_inlet,
# #         'A_throat':   A_throat,
# #         'A_exit':     A_exit,
# #         'bell':       bell,
# #         'bx':         bx,
# #         'y_center':   y_center,
# #         'station_x': {
# #             0:   float(fore[0]),
# #             2:   float(throat_x0),          # NEW station: throat
# #             3:   float(duct_x0),            # Combustor face (was throat)
# #             4:   float(duct_x1),
# #             9:   float(bx[-1]),
# #         },
# #         'station_labels': {
# #             0: 'Freestream',
# #             2: 'Throat',
# #             3: 'Combustor face',
# #             4: 'Combustor exit',
# #             9: 'Nozzle exit',
# #         },
# #     }
#
#
# # ---- Flowpath figure: draw diffuser walls + terminal shock ----------------
# # Wherever the flowpath is drawn (e.g. fig_flowpath / fig_axial_properties),
# # add after the ramp/throat polylines and before the combustor rectangle:
# #
# #     ax.plot(layout['diff_upper'][:, 0], layout['diff_upper'][:, 1],
# #             color='k', lw=1.5)
# #     ax.plot(layout['diff_lower'][:, 0], layout['diff_lower'][:, 1],
# #             color='k', lw=1.5)
# #     ax.fill_between(layout['diff_upper'][:, 0],
# #                     layout['diff_lower'][:, 1],
# #                     layout['diff_upper'][:, 1],
# #                     color='tab:blue', alpha=0.06,
# #                     label='Subsonic diffuser')
# #
# #     # Terminal shock — pulled from the converged cycle case (design_cycle
# #     # must carry x_terminal_shock from DiffuserTerminalShock output).
# #     x_s = design_cycle.get('x_terminal_shock')
# #     if x_s is not None:
# #         y_top = np.interp(x_s, layout['diff_upper'][:, 0],
# #                                 layout['diff_upper'][:, 1])
# #         y_bot = np.interp(x_s, layout['diff_lower'][:, 0],
# #                                 layout['diff_lower'][:, 1])
# #         ax.plot([x_s, x_s], [y_bot, y_top], color='red', lw=2,
# #                 linestyle='--', label='Terminal normal shock')
#
#
# # ---- fig_inlet_pt_vs_mach update -------------------------------------------
# # evaluate_fixed_geometry_at_condition now needs p_back. Two reasonable
# # options; pick one per figure:
# #
# #   (i) Design-point back pressure: hold p_back fixed at the converged
# #       combustor Ps from the design cycle. Good for "how does recovery
# #       drift off-design at the same combustor operating point?".
# #
# #  (ii) Back-pressure sweep: add a second axis / parametric line family
# #       over p_back / Pt_after_cowl in [p_swallow_fraction, p_expel_fraction]
# #       to visualize the operating envelope.
# #
# # Snippet for (i):
# #     p_back_design = float(design_cycle['combustor_Ps_Pa'])
# #     for M in mach_vals:
# #         case = _inlet2.evaluate_fixed_geometry_at_condition(
# #             design, M0=M, altitude_m=INLET_DESIGN_ALT_M,
# #             alpha_deg=0.0, p_back=p_back_design,
# #         )
# #         if case.get('success', False):
# #             ys.append(case['pt_frac_after_terminal_shock'])
# #         else:
# #             ys.append(np.nan)   # unstart -> gap in the curve
# #
# # New figure (recommended): shock position vs p_back at design Mach.
# # def fig_shock_position_vs_backpressure(design, design_cycle):
# #     Pt_ac = float(design_cycle['Pt_after_cowl_Pa'])
# #     pr = np.linspace(0.05, 0.95, 60)     # p_back / Pt_after_cowl
# #     xs_list, pt_list, flag_list = [], [], []
# #     for r in pr:
# #         case = _inlet2.evaluate_fixed_geometry_at_condition(
# #             design, M0=INLET_DESIGN_M0, altitude_m=INLET_DESIGN_ALT_M,
# #             alpha_deg=INLET_DESIGN_ALPHA_DEG, p_back=r * Pt_ac,
# #         )
# #         if case.get('success', False):
# #             xs_list.append(case['x_terminal_shock'])
# #             pt_list.append(case['pt_frac_after_terminal_shock'])
# #             flag_list.append(0)
# #         else:
# #             xs_list.append(np.nan)
# #             pt_list.append(np.nan)
# #             flag_list.append(+1 if case.get('status') == 'expelled' else -1)
# #     # Plot xs_list and pt_list vs pr; shade regions where flag != 0 as
# #     # "unstart (expelled)" / "unstart (swallowed)".
# #     _save(fig, 'fig07_shock_position_vs_backpressure')


'''