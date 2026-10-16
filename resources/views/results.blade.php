<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <title>Laravel</title>

        <!-- Fonts -->
        <link rel="preconnect" href="https://fonts.bunny.net">
        <link href="https://fonts.bunny.net/css?family=instrument-sans:400,500,600" rel="stylesheet" />
        @if (file_exists(public_path('build/manifest.json')) || file_exists(public_path('hot')))
            @vite(['resources/css/app.css', 'resources/js/app.js'])
            @endif
    </head>
    <body class="bg-[#FDFDFC] dark:bg-[#0a0a0a] text-[#1b1b18] flex p-6 lg:p-8 items-center lg:justify-center min-h-screen flex-col">

<div class="container p-4 mx-auto">
    @foreach($detailed_matches as $match)
    <div class="p-6 mb-8 bg-white rounded-lg shadow-lg">
        <div class="flex justify-between items-center mb-4">
            <h3 class="text-xl font-bold">{{ $match['pattern_name'] }}</h3>
            <span class="px-4 py-2 rounded-full {{ getScoreColorClass($match['similarity_score']) }}">
                {{ number_format($match['similarity_score'] * 100, 1) }}% Match
            </span>
        </div>

        <div class="mb-4">
            <p class="text-gray-700">{{ $match['description'] }}</p>
        </div>

        <!-- Pattern Explanation -->
        <div class="mt-6">
            <h4 class="mb-3 text-lg font-semibold">Why this pattern matches:</h4>

            <!-- Key Phrases Analysis -->
            <div class="mb-4">
                <h5 class="mb-2 font-medium text-md">Key Aspects Found:</h5>
                <div class="flex flex-wrap gap-2">
                    @foreach($match['explanation']['key_phrases'] as $phrase)
                        <span class="px-3 py-1 text-sm rounded-full {{ getScoreColorClass($phrase['relevance']) }}">
                            {{ $phrase['text'] }}
                            <span class="ml-1 text-xs">({{ number_format($phrase['relevance'] * 100, 1) }}%)</span>
                        </span>
                    @endforeach
                </div>
            </div>

            <!-- Supporting Elements -->
            <div class="mt-4">
                <h5 class="mb-2 font-medium text-md">Supporting Elements Detected:</h5>
                <div class="grid grid-cols-1 gap-2 md:grid-cols-2">
                    @foreach($match['explanation']['supporting_elements'] as $element)
                        <div class="flex items-center p-2 bg-gray-50 rounded">
                            <span class="w-2 h-2 rounded-full mr-2
                                {{ $element['contribution'] === 'high' ? 'bg-green-500' :
                                   ($element['contribution'] === 'medium' ? 'bg-yellow-500' : 'bg-blue-500') }}">
                            </span>
                            <span class="text-sm">
                                {{ $element['element'] }}
                                <span class="text-xs text-gray-500">
                                    ({{ number_format($element['confidence'] * 100, 1) }}%)
                                </span>
                            </span>
                        </div>
                    @endforeach
                </div>
            </div>
        </div>
    </div>

    <!-- <div class="mt-6">
        <h4 class="mb-3 text-lg font-semibold">Visual Explanation</h4>

        <div class="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div class="p-4 rounded-lg border">
                <h5 class="mb-2 font-medium">Pattern Attention Map</h5>
                <img src="{{ $match['explanation']['visualizations']['attention_map'] }}"
                     alt="Attention heatmap for {{ $match['pattern_name'] }}"
                     class="rounded shadow-sm">
                <p class="mt-2 text-sm text-gray-600">
                    Highlighted regions show areas that strongly match this pattern
                </p>
            </div>

            <div class="p-4 rounded-lg border">
                <h5 class="mb-2 font-medium">Key Regions</h5>
                <img src="{{ $match['explanation']['visualizations']['regions_of_interest'] }}"
                     alt="Regions of interest for {{ $match['pattern_name'] }}"
                     class="rounded shadow-sm">
                <p class="mt-2 text-sm text-gray-600">
                    Boxes indicate specific areas that exemplify this pattern
                </p>
            </div>
        </div>
    </div> -->
    @endforeach
</div>


    </body>
</html>
@php

function getScoreColorClass($score) {
    if ($score >= 0.8) {
        return 'bg-green-100 text-green-800';
    } elseif ($score >= 0.6) {
        return 'bg-blue-100 text-blue-800';
    } elseif ($score >= 0.4) {
        return 'bg-yellow-100 text-yellow-800';
    } else {
        return 'bg-red-100 text-red-800';
    }
}
@endphp
