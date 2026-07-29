import { Injectable, inject, signal } from '@angular/core';
import { ReportsService } from './reports.service';
import {
  OperationalMetrics,
  UtilizationReport,
  FinancialReport,
  AuditTrail
} from './reports.model';

export type ReportTabId = 'operational' | 'utilization' | 'financial' | 'audit';

export interface ReportTab {
  id: ReportTabId;
  name: string;
  icon: string;
}

@Injectable()
export class ReportsPageState {
  private readonly reportsService = inject(ReportsService);

  readonly activeTab = signal<ReportTabId>('operational');

  readonly operationalPeriod = signal<string>('week');
  readonly utilizationPeriod = signal<string>('month');
  readonly financialPeriod = signal<string>('month');

  readonly auditFilter = signal<string>('all');
  readonly auditDateFrom = signal<string>('');
  readonly auditDateTo = signal<string>('');

  readonly operationalMetrics = signal<OperationalMetrics | null>(null);
  readonly utilizationReport = signal<UtilizationReport | null>(null);
  readonly financialReport = signal<FinancialReport | null>(null);
  readonly auditTrails = signal<AuditTrail | null>(null);

  readonly reportTabs: ReportTab[] = [
    {
      id: 'operational',
      name: 'Operational Metrics',
      icon: 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6'
    },
    {
      id: 'utilization',
      name: 'Utilization Reports',
      icon: 'M4 6h16M4 10h16M4 14h16M4 18h16'
    },
    {
      id: 'financial',
      name: 'Financial Reports',
      icon: 'M13 10V3L4 14h7v7l9-11h-7z'
    },
    {
      id: 'audit',
      name: 'Audit Trails',
      icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z'
    }
  ];

  loadAll(): void {
    this.loadOperationalMetrics();
    this.loadUtilizationReports();
    this.loadFinancialReports();
    this.loadAuditTrails();
  }

  setActiveTab(tabId: ReportTabId): void {
    this.activeTab.set(tabId);
  }

  loadOperationalMetrics(): void {
    this.reportsService
      .getOperationalMetrics(this.operationalPeriod())
      .subscribe(metrics => this.operationalMetrics.set(metrics));
  }

  loadUtilizationReports(): void {
    this.reportsService
      .getUtilizationReport(this.utilizationPeriod())
      .subscribe(report => this.utilizationReport.set(report));
  }

  loadFinancialReports(): void {
    this.reportsService
      .getFinancialReport(this.financialPeriod())
      .subscribe(report => this.financialReport.set(report));
  }

  loadAuditTrails(): void {
    this.reportsService
      .getAuditTrails(this.auditFilter(), this.auditDateFrom(), this.auditDateTo())
      .subscribe(trails => this.auditTrails.set(trails));
  }

  exportReport(type: ReportTabId): void {
    this.reportsService.exportReport(type, this.getReportPeriod(type)).subscribe(blob => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${type}-report-${new Date().toISOString().split('T')[0]}.pdf`;
      a.click();
      window.URL.revokeObjectURL(url);
    });
  }

  getReportPeriod(type: ReportTabId): string {
    switch (type) {
      case 'operational':
        return this.operationalPeriod();
      case 'utilization':
        return this.utilizationPeriod();
      case 'financial':
        return this.financialPeriod();
      default:
        return 'month';
    }
  }

  getTabClass(tabId: ReportTabId): string {
    return tabId === this.activeTab()
      ? 'border-primary-500 text-primary-600'
      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300';
  }

  getUtilizationBarClass(utilization: number): string {
    if (utilization >= 90) return 'bg-error-500';
    if (utilization >= 75) return 'bg-warning-500';
    if (utilization >= 50) return 'bg-primary-500';
    return 'bg-success-500';
  }

  getUtilizationStatusClass(utilization: number): string {
    if (utilization >= 90) return 'bg-error-100 text-error-800';
    if (utilization >= 75) return 'bg-warning-100 text-warning-800';
    if (utilization >= 50) return 'bg-primary-100 text-primary-800';
    return 'bg-success-100 text-success-800';
  }

  getUtilizationStatus(utilization: number): string {
    if (utilization >= 90) return 'Critical';
    if (utilization >= 75) return 'High';
    if (utilization >= 50) return 'Moderate';
    return 'Low';
  }

  getBillingStatusClass(status: string): string {
    switch (status) {
      case 'paid':
        return 'bg-success-100 text-success-800';
      case 'pending':
        return 'bg-warning-100 text-warning-800';
      case 'overdue':
        return 'bg-error-100 text-error-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  getActionTypeClass(actionType: string): string {
    switch (actionType) {
      case 'create':
        return 'bg-success-100 text-success-800';
      case 'update':
        return 'bg-primary-100 text-primary-800';
      case 'delete':
        return 'bg-error-100 text-error-800';
      case 'login':
        return 'bg-secondary-100 text-secondary-800';
      case 'logout':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  getEventStatusClass(status: string): string {
    switch (status) {
      case 'success':
        return 'bg-success-100 text-success-800';
      case 'failed':
        return 'bg-error-100 text-error-800';
      case 'warning':
        return 'bg-warning-100 text-warning-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  formatCurrency(value: number | undefined): string {
    if (!value) return '0';
    return new Intl.NumberFormat('en-US').format(value);
  }

  getChartXPosition(index: number): number {
    const chartWidth = 440;
    const dataPoints = this.operationalMetrics()?.dailyThroughputTrend?.length || 1;
    const spacing = chartWidth / (dataPoints - 1);
    return 40 + index * spacing;
  }

  getChartYPosition(value: number): number {
    const trend = this.operationalMetrics()?.dailyThroughputTrend || [];
    const maxValue = Math.max(...(trend.map(d => d.value) || [100]));
    const minValue = Math.min(...(trend.map(d => d.value) || [0]));
    const range = maxValue - minValue || 1;
    const chartHeight = 140;
    const normalizedValue = (value - minValue) / range;
    return 160 - normalizedValue * chartHeight;
  }

  getChartLinePath(): string {
    const trend = this.operationalMetrics()?.dailyThroughputTrend;
    if (!trend) return '';

    return trend
      .map((item, index) => {
        const x = this.getChartXPosition(index);
        const y = this.getChartYPosition(item.value);
        return `${x},${y}`;
      })
      .join(' ');
  }

  getChartDataPoints(): { x: number; y: number; value: number }[] {
    const trend = this.operationalMetrics()?.dailyThroughputTrend;
    if (!trend) return [];

    return trend.map((item, index) => ({
      x: this.getChartXPosition(index),
      y: this.getChartYPosition(item.value),
      value: item.value
    }));
  }

  getChartYAxisLabels(): { value: number; y: number }[] {
    const trend = this.operationalMetrics()?.dailyThroughputTrend;
    if (!trend || trend.length === 0) return [];

    const maxValue = Math.max(...trend.map(d => d.value));
    const minValue = Math.min(...trend.map(d => d.value));
    const range = maxValue - minValue || 1;
    const step = range / 4;

    return [0, 1, 2, 3, 4].map(i => ({
      value: Math.round(maxValue - i * step),
      y: 20 + i * 35
    }));
  }
}
