from futures_backtester.data import make_sample_panel, panel_to_frame


def test_sample_panel_shapes_and_metadata() -> None:
    panel = make_sample_panel(days=300, contracts=12, seed=1)

    assert panel.prices.shape == (300, 12)
    assert panel.carry.shape == (300, 12)
    assert panel.roll_cost_bps.shape == (300, 12)
    assert panel.commission_bps.shape == (12,)
    assert panel.sectors.notna().all()


def test_panel_to_frame_contains_expected_fields() -> None:
    panel = make_sample_panel(days=40, contracts=5, seed=2)
    frame = panel_to_frame(panel)

    expected_columns = {
        "date",
        "contract",
        "price",
        "carry",
        "roll_cost_bps",
        "sector",
        "commission_bps",
    }
    assert expected_columns <= set(frame.columns)
    assert len(frame) == 40 * 5
