<script>
	import ArcChart from '$lib/components/app/arcChart.svelte';
	import * as Card from '$lib/components/ui/card/index.js';
	import { File, Download, ArrowLeft } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button/index.js';

	let {
		detectionConfidence = 0,
		dogBreed = '',
		selectedFile = $bindable(),
		currentStep = $bindable()
	} = $props();

	// Reset so it shows upload card and removes the uploaded file
	function reset() {
		currentStep = 1;
		selectedFile = null;
	}
</script>

<!-- Card component -->
<Card.Root class="w-11/12 md:w-4/12">
	<Card.Content>
		<Button class="p4" variant="outline" size="icon" onclick={reset}><ArrowLeft /></Button>

		<!-- Arc chart component -->
		<div class="h-60 p-4">
			<ArcChart {detectionConfidence} {dogBreed} />
		</div>

		<!-- Medical disclaimer -->
		<div class="mt-7 w-full gap-y-2 text-pretty rounded-2xl bg-muted p-4">
			<p class="text-sm text-muted-foreground">
				Based on the image provided, the AI analysis suggests a <span class="font-bold"
					>{detectionConfidence}%</span
				> likelihood of it being a <span class="font-bold">{dogBreed}</span>. Please note this is only an AI-based estimate and not a definitive identification. Dog breeds can sometimes be difficult to identify from images alone, especially with mixed breeds or unusual coloring.
			</p>
		</div>
	</Card.Content>
</Card.Root>
