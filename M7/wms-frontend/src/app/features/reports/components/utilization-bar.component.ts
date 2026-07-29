import { Component, input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-utilization-bar',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="space-y-2">
      <div class="flex justify-between">
        <span class="text-sm text-gray-600 dark:text-gray-400">{{ label() }}</span>
        <span class="text-sm font-medium text-gray-900 dark:text-white">{{ value() }}%</span>
      </div>
      <div class="w-full bg-gray-200 dark:bg-dark-700 rounded-full h-2">
        <div
          [class]="barClass()"
          class="h-2 rounded-full transition-all"
          [style.width.%]="value()"
        ></div>
      </div>
      @if (details()) {
        <div class="text-xs text-gray-500 dark:text-gray-400">{{ details() }}</div>
      }
    </div>
  `
})
export class UtilizationBarComponent {
  label = input.required<string>();
  value = input.required<number>();
  details = input<string>();
  barClass = input<string>('bg-primary-500');
}
