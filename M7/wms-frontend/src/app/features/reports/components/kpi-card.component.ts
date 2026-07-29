import { Component, computed, input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-kpi-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="card p-6">
      <div class="flex items-center">
        <div class="p-2 rounded-lg" [class]="bgClass()">
          <svg
            class="h-6 w-6"
            [class]="iconClass()"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              [attr.d]="icon()"
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
            />
          </svg>
        </div>
        <div class="ml-4">
          <p class="text-sm font-medium text-gray-600 dark:text-gray-400">{{ label() }}</p>
          <p class="text-2xl font-semibold text-gray-900 dark:text-white">{{ value() }}</p>
          @if (sublabel()) {
            <p class="text-xs text-gray-500">{{ sublabel() }}</p>
          }
        </div>
      </div>
    </div>
  `
})
export class KpiCardComponent {
  icon = input.required<string>();
  label = input.required<string>();
  value = input.required<string>();
  sublabel = input<string>();
  color = input<'primary' | 'success' | 'warning' | 'error' | 'secondary'>('primary');

  protected readonly bgClass = computed(() => `bg-${this.color()}-100`);
  protected readonly iconClass = computed(() => `text-${this.color()}-600`);
}
