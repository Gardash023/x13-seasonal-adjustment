import numpy as np
import pandas as pd
from x13_seasonal_adjustment import X13SeasonalAdjustment

def test_perf_smoke(benchmark):
	# Small synthetic dataset for quick CI run
	dates = pd.date_range('2010-01-01', periods=60, freq='M')
	data = pd.Series(
		100 + np.sin(2 * np.pi * np.arange(60) / 12) + np.random.normal(0, 0.1, 60),
		index=dates,
	)
	x13 = X13SeasonalAdjustment(freq='M')

	def run():
		x13.fit_transform(data)

	benchmark(run)
