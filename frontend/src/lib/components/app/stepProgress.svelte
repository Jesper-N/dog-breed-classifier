<script>
	import { Check } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Separator } from '$lib/components/ui/separator/index.js';

	let { currentStep = 1 } = $props();
</script>

<!--  Display a two-step progress indicator showing current stage (Upload/Results) with checkmarks for completed steps-->
<div class="pointer-events-none flex w-11/12 items-center justify-center gap-2 md:w-4/12">
	{#each ['Upload file', 'Results'] as step, i}
		{#if i > 0}
			<Separator class="w-8" />
		{/if}
		<div class="inline-flex items-center gap-x-2">
			<Button disabled={currentStep < i + 1} class="size-8">
				{#if currentStep > i + 1 || (currentStep === 2 && i === 1)}
					<Check />
				{:else}
					{i + 1}
				{/if}
			</Button>
			<p class={currentStep === i + 1 ? 'font-bold' : ''}>{step}</p>
		</div>
	{/each}
</div>
