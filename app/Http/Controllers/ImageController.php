<?php

namespace App\Http\Controllers;

use App\Models\UploadedImage;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class ImageController extends Controller
{
    public function index()
    {
        return view('analyze');
    }

    public function analyze(Request $request)
    {
        $request->validate(['image' => 'required|image|max:2048']);

        $path = $request->file('image')->store('uploads', 'public');

        $image = UploadedImage::create(['path' => $path]);

        // Call Python ML backend
        $response = Http::attach('image', file_get_contents($request->image), 'image.jpg')
            ->acceptJson()
            ->post('http://localhost:8003/detect-patterns');

        $detectedPatterns = $response->json();

        // Process visualizations for each pattern
        // Convert to collection and process
        $detailedMatches = collect($detectedPatterns['detailed_matches'])->map(function ($match) {
            if (isset($match['explanation']['visualizations'])) {
                $match['explanation']['visualizations'] = collect($match['explanation']['visualizations'])
                    ->map(fn ($path) => 'http://localhost:8003'.$path)
                    ->all();
            }

            return $match;
        });

        return view('results', [
            'detailed_matches' => $detailedMatches->all(),
            'analysis_metadata' => $detectedPatterns['analysis_metadata'],
            'steps' => collect($centersData['steps'] ?? [])
                ->map(fn ($path) => 'http://localhost:8003'.$path)
                ->all(),
            'original_image_path' => asset('storage/'.$path),
            'image_id' => $image->id,
        ]);
    }
}
