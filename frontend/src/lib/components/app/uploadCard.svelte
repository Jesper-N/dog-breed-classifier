<script>
	import { FileUp, File } from 'lucide-svelte';
	import { toast } from 'svelte-sonner';
	import * as Card from '$lib/components/ui/card/index.js';
	import { Progress } from '$lib/components/ui/progress/index.js';

	// Image spec requirements
	const MAX_SIZE_MB = 60;
	const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;
	const ALLOWED_TYPES = ['image/jpeg', 'image/png'];

	let dragOver = $state(false);
	let loading = $state(false);
	let uploadProgress = $state(0);

	let {
		currentStep = $bindable(),
		selectedFile = $bindable(),
		detectionConfidence = $bindable(),
		dogBreed = $bindable()
	} = $props();

	// Convert uploaded file to base64 string
	function convertToBase64(file) {
		return new Promise((resolve, reject) => {
			const reader = new FileReader();
			reader.onload = () => resolve(btoa(reader.result));
			reader.onerror = reject;
			reader.readAsBinaryString(file);
		});
	}

	// Upload base64 encoded image to API. With progress tracking
	async function uploadImage(base64) {
		const xhr = new XMLHttpRequest();

		return new Promise((resolve, reject) => {
			xhr.upload.onprogress = (event) => {
				if (event.lengthComputable) {
					uploadProgress = Math.round((event.loaded / event.total) * 100);
				}
			};

			xhr.onload = () => {
				if (xhr.status === 200) {
					resolve(JSON.parse(xhr.response));
				} else {
					reject(new Error('Upload failed'));
				}
			};

			xhr.onerror = () => reject(new Error('Upload failed'));

			xhr.open('POST', '/api/guess-breed');
			xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
			xhr.send(
				new URLSearchParams({
					base64_file: base64
				}).toString()
			);
		});
	}

	// Process and validate uploaded image file, convert to base64, send to server for dog breed prediction
	async function handleFile(file) {
		if (!file) return;

		if (!ALLOWED_TYPES.includes(file.type)) {
			toast.error('Only JPEG and PNG files are allowed');
			return;
		}

		if (file.size > MAX_SIZE_BYTES) {
			toast.error(`File must be under ${MAX_SIZE_MB}MB`);
			return;
		}

		try {
			loading = true;
			uploadProgress = 0;
			selectedFile = file;

			const base64 = await convertToBase64(file);
			const { percent, dog_breed } = await uploadImage(base64);
			detectionConfidence = percent;
			dogBreed = dog_breed;
			currentStep = 2;
		} catch (error) {
			console.error('Upload failed:', error);
			toast.error('Failed to process image');
			currentStep = 1;
			selectedFile = null;
		} finally {
			loading = false;
			uploadProgress = 0;
		}
	}
</script>

<!-- Upload card -->
<Card.Root class="w-11/12 md:w-4/12">
	<Card.Content>
		<!-- Drag and drop section -->
		<label
			class="group relative flex h-56 w-full flex-col items-center justify-center gap-y-2 rounded-2xl border-[2.3px] border-dashed transition-colors {dragOver
				? 'border-primary bg-primary/5'
				: 'border-black hover:border-primary hover:bg-primary/5 dark:border-muted'}"
			ondragenter={(e) => {
				e.preventDefault();
				dragOver = true;
			}}
			ondragleave={(e) => {
				e.preventDefault();
				dragOver = false;
			}}
			ondragover={(e) => e.preventDefault()}
			ondrop={(e) => {
				e.preventDefault();
				dragOver = false;
				handleFile(e.dataTransfer?.files[0]);
			}}
		>
			<input
				type="file"
				accept={ALLOWED_TYPES.join(',')}
				capture="environment"
				class="hidden"
				onchange={(e) => handleFile(e.currentTarget.files?.[0])}
				disabled={loading}
			/>

			<div class="pointer-events-none flex flex-col items-center gap-y-4">
				<FileUp
					class="transition-colors {dragOver ? 'text-primary' : 'group-hover:text-primary'}"
					size={70}
					strokeWidth={0.5}
				/>
				<p>
					{#if loading}
						Processing...
					{:else}
						Drag and Drop file here or
						<span class="font-bold">Choose file</span>
					{/if}
				</p>
			</div>
		</label>

		<!-- Display image file requirements -->
		<div class="mt-3 flex justify-between text-sm text-muted-foreground">
			<p>Supported formats: JPEG, PNG</p>
			<p>Maximum size: {MAX_SIZE_MB} MB</p>
		</div>

		<!-- Show preview/specs of the uploaded file -->
		{#if selectedFile}
			<div class="mt-10 flex w-full flex-col gap-y-2 rounded-2xl bg-muted p-4">
				<div class="flex items-center gap-x-2">
					<File size={40} strokeWidth={1} />
					<div class="grid">
						<p class="font-bold">{selectedFile.name}</p>
						<p class="text-sm text-muted-foreground">
							{Math.round(selectedFile.size / (1024 * 1024))} MB
						</p>
					</div>
				</div>
				<!-- Show progress of sending API request with image to the server. Useful for slow internet -->
				{#if loading}
					<Progress class="bg-primary" value={uploadProgress} />
				{/if}
			</div>
		{/if}
	</Card.Content>
</Card.Root>
