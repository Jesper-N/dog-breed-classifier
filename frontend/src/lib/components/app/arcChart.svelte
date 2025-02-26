<script>
	import { Arc, Chart, Svg, Text } from 'layerchart';
	import { cls } from '@layerstack/tailwind';
	import { cubicInOut } from 'svelte/easing';

	let { detectionConfidence = 0 } = $props();
</script>

<!-- Arc chart that shows the AI detection confidence in percentage  -->
<div class="relative grid h-full place-items-center">
	<Chart>
		<Svg center>
			<Arc
				initialValue={0}
				value={detectionConfidence}
				innerRadius={-20}
				cornerRadius={4}
				class="fill-primary"
				track={{ class: 'fill-zinc-500/10' }}
				tweened={{ duration: 700, easing: cubicInOut }}
			/>
		</Svg>
	</Chart>
	<div class="absolute">
		<span class="text-6xl tabular-nums dark:text-white">
			{Math.round(detectionConfidence ?? 0)}
		</span>
	</div>
</div>
