import { Component, input, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DropdownComponent } from '../../../ui-library/Dropdown.component';
import { Heading3Component, SectionHeadingComponent } from '../../../ui-library/Typography/Typography.component';
import { OperationalMetrics } from '../reports.model';
import { KpiCardComponent } from './kpi-card.component';

@Component({
  selector: 'app-operational-metrics',
  standalone: true,
  imports: [
    CommonModule,
    DropdownComponent,
    Heading3Component,
    SectionHeadingComponent,
    KpiCardComponent
  ],
  template: `
    <div class="p-6">
      <div class="flex justify-between items-center mb-6">
        <ui-heading3>Operational Metrics & KPIs</ui-heading3>
        <div class="flex space-x-3">
          <ui-dropdown
            label="Period"
            [options]="periodOptions"
            [value]="period()"
            (valueChange)="periodChange.emit($event)"
          />
          <button (click)="export.emit()" class="btn btn-secondary">
            <svg
              class="h-4 w-4 mr-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            Export
          </button>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <app-kpi-card
          icon="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
          label="Throughput"
          [value]="(metrics()?.throughput || 0).toString()"
          sublabel="items/hour"
          color="primary"
        />
        <app-kpi-card
          icon="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
          label="Order Accuracy"
          [value]="(metrics()?.orderAccuracy || 0) + '%'"
          sublabel="accuracy rate"
          color="success"
        />
        <app-kpi-card
          icon="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          label="Avg Processing Time"
          [value]="(metrics()?.avgProcessingTime || 0).toString()"
          sublabel="minutes"
          color="warning"
        />
        <app-kpi-card
          icon="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          label="Error Rate"
          [value]="(metrics()?.errorRate || 0) + '%'"
          sublabel="error rate"
          color="error"
        />
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="card p-6">
          <ui-section-heading>Daily Throughput Trend</ui-section-heading>
          <div class="h-64 bg-gray-50 dark:bg-dark-700 rounded-lg p-4">
            @if (metrics()?.dailyThroughputTrend) {
              <svg
                class="w-full h-full"
                viewBox="0 0 500 200"
                preserveAspectRatio="xMidYMid meet"
              >
                <line x1="40" y1="20" x2="40" y2="160" stroke="#e5e7eb" stroke-width="2" />
                <line x1="40" y1="160" x2="480" y2="160" stroke="#e5e7eb" stroke-width="2" />

                @for (line of [0, 1, 2, 3, 4]; track line) {
                  <line
                    [attr.x1]="40"
                    [attr.y1]="20 + line * 35"
                    [attr.x2]="480"
                    [attr.y2]="20 + line * 35"
                    stroke="#f3f4f6"
                    stroke-width="1"
                    stroke-dasharray="5,5"
                  />
                }

                @for (label of yAxisLabels(); track label.value) {
                  <text
                    [attr.x]="30"
                    [attr.y]="label.y + 5"
                    class="text-xs fill-gray-600 dark:fill-gray-400"
                    text-anchor="end"
                  >
                    {{ label.value }}
                  </text>
                }

                <polyline
                  [attr.points]="linePath()"
                  fill="none"
                  stroke="#3B82F6"
                  stroke-width="3"
                  stroke-linejoin="round"
                />

                @for (point of dataPoints(); track point.x) {
                  <circle
                    [attr.cx]="point.x"
                    [attr.cy]="point.y"
                    r="5"
                    fill="#3B82F6"
                    class="hover:fill-primary-700 cursor-pointer"
                  />
                }

                @for (item of metrics()?.dailyThroughputTrend; track item.date; let i = $index) {
                  <text
                    [attr.x]="xPosition(i)"
                    [attr.y]="180"
                    class="text-xs fill-gray-600 dark:fill-gray-400"
                    text-anchor="middle"
                  >
                    {{ item.date }}
                  </text>
                }

                @for (point of dataPoints(); track point.x) {
                  <text
                    [attr.x]="point.x"
                    [attr.y]="point.y - 10"
                    class="text-xs fill-gray-900 dark:fill-white font-medium"
                    text-anchor="middle"
                  >
                    {{ point.value }}
                  </text>
                }
              </svg>
            } @else {
              <div class="flex items-center justify-center h-full">
                <p class="text-gray-500 dark:text-gray-400">No chart data available</p>
              </div>
            }
          </div>
        </div>

        <div class="card p-6">
          <ui-section-heading>Order Processing Performance</ui-section-heading>
          <div class="space-y-4">
            @for (metric of metrics()?.detailedMetrics; track metric.name) {
              <div class="flex items-center justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-400">{{ metric.name }}</span>
                <div class="flex items-center space-x-2">
                  <div class="w-24 bg-gray-200 dark:bg-dark-700 rounded-full h-2">
                    <div
                      class="bg-primary-600 h-2 rounded-full"
                      [style.width.%]="metric.value"
                    ></div>
                  </div>
                  <span class="text-sm font-medium text-gray-900 dark:text-white">{{ metric.value }}%</span>
                </div>
              </div>
            }
          </div>
        </div>
      </div>
    </div>
  `
})
export class OperationalMetricsComponent {
  metrics = input<OperationalMetrics | null>(null);
  period = input<string>('week');
  periodChange = output<string>();
  export = output<void>();

  readonly periodOptions = [
    { value: 'today', label: 'Today' },
    { value: 'week', label: 'This Week' },
    { value: 'month', label: 'This Month' },
    { value: 'quarter', label: 'This Quarter' }
  ];

  private get trend() {
    return this.metrics()?.dailyThroughputTrend ?? [];
  }

  protected xPosition(index: number): number {
    const chartWidth = 440;
    const dataPoints = this.trend.length || 1;
    const spacing = chartWidth / (dataPoints - 1);
    return 40 + index * spacing;
  }

  private yPosition(value: number): number {
    const maxValue = Math.max(...(this.trend.map(d => d.value) || [100]));
    const minValue = Math.min(...(this.trend.map(d => d.value) || [0]));
    const range = maxValue - minValue || 1;
    const chartHeight = 140;
    const normalizedValue = (value - minValue) / range;
    return 160 - normalizedValue * chartHeight;
  }

  protected linePath(): string {
    if (!this.trend.length) return '';
    return this.trend
      .map((item, index) => `${this.xPosition(index)},${this.yPosition(item.value)}`)
      .join(' ');
  }

  protected dataPoints(): { x: number; y: number; value: number }[] {
    if (!this.trend.length) return [];
    return this.trend.map((item, index) => ({
      x: this.xPosition(index),
      y: this.yPosition(item.value),
      value: item.value
    }));
  }

  protected yAxisLabels(): { value: number; y: number }[] {
    if (!this.trend.length) return [];
    const maxValue = Math.max(...this.trend.map(d => d.value));
    const minValue = Math.min(...this.trend.map(d => d.value));
    const range = maxValue - minValue || 1;
    const step = range / 4;
    return [0, 1, 2, 3, 4].map(i => ({
      value: Math.round(maxValue - i * step),
      y: 20 + i * 35
    }));
  }
}
