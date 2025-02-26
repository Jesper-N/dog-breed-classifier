<script>
	import '../app.css';
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from '$lib/components/ui/sonner/index.js';
	import Topbar from '$lib/components/app/topbar.svelte';

	let { children } = $props();
</script>

<!-- For showing notifications/alerts -->
<Toaster />

<!-- For theme changing -->
<ModeWatcher />

<!-- Render page with topbar -->
<div class="flex h-full min-h-screen flex-col">
	<Topbar />
	{@render children()}
</div>
