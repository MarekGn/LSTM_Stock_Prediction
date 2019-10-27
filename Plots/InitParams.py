import matplotlib

# fig_width 80mm (here size in inches)
fig_width = 3.34
fig_height = 2.2
linewidths=0.75
tickwidths=0.8*linewidths
tickminor=1.5
tickmajor=2*tickminor
params = {
			'font.size':11,
			'axes.labelsize': 10, 
			'axes.titlesize': 10,
			'legend.fontsize': 8,
			'xtick.labelsize': 8.5,
			'ytick.labelsize': 8.5,
			
			'font.family': 'serif',
			# 'mathtext.fontset': 'stixsans',
			# 'mathtext.rm': 'sans',
			# 'mathtext.bf': 'serif:sans',
			# 'mathtext.it': 'serif:sans',
			# 'font.weight': 'medium',

			'xtick.direction': 'in',
			'ytick.direction': 'in',
			'ytick.major.width': tickwidths,
			'xtick.major.width': tickwidths,
			'ytick.minor.width': tickwidths,
			'xtick.minor.width': tickwidths,
			'ytick.minor.size': tickminor,
			'xtick.minor.size': tickminor,
			'ytick.major.size': tickmajor,
			'xtick.major.size': tickmajor,
			'ytick.right': True,
			'xtick.top': True,
			'text.usetex': True,
			'figure.figsize': [fig_width, fig_height],

			'axes.linewidth': linewidths,
			'lines.linewidth': linewidths,
			'lines.markeredgewidth': linewidths,
			'lines.markersize': 4,
			'legend.frameon': True,
			'legend.handlelength': 1,
			'ps.usedistiller': 'xpdf',
}
matplotlib.rcParams.update(params)
# matplotlib.rcParams['text.latex.preamble'] = [
#        r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#        r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
#        r'\usepackage{helvet}',    # set the normal font here
#        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#        r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
# ]  