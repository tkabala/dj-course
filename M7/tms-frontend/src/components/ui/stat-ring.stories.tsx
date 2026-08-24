import React from 'react';
import { Meta, StoryObj } from '@storybook/react-vite';
import { StatRing } from './stat-ring';
import { Truck, Clock, ShieldCheck, AlertTriangle } from 'lucide-react';

const meta: Meta<typeof StatRing> = {
  title: 'UI/StatRing',
  component: StatRing,
  argTypes: {
    variant: {
      control: 'select',
      options: ['blue', 'green', 'purple', 'orange', 'red'],
    },
    value: { control: { type: 'range', min: 0, max: 100 } },
  },
  tags: ['autodocs'],
};

export default meta;

type Story = StoryObj<typeof StatRing>;

export const Default: Story = {
  args: {
    value: 72,
    displayValue: '72',
    label: 'Total Deliveries',
  },
};

export const OnTimeRate: Story = {
  args: {
    value: 98,
    displayValue: '98%',
    label: 'On-Time Rate',
    variant: 'green',
    icon: <Clock className="h-4 w-4" />,
  },
};

export const SafetyScore: Story = {
  args: {
    value: 91,
    displayValue: '91',
    label: 'Safety Score',
    variant: 'purple',
    icon: <ShieldCheck className="h-4 w-4" />,
  },
};

export const LowValue: Story = {
  args: {
    value: 20,
    displayValue: '2',
    label: 'Incidents',
    variant: 'orange',
    icon: <AlertTriangle className="h-4 w-4" />,
  },
};

export const Empty: Story = {
  args: {
    value: 0,
    displayValue: '0',
    label: 'No data yet',
    variant: 'red',
  },
};

export const Group: Story = {
  render: () => (
    <div className="flex flex-wrap gap-8">
      <StatRing value={72} displayValue="72" label="Total Deliveries" variant="blue" icon={<Truck className="h-4 w-4" />} />
      <StatRing value={98} displayValue="98%" label="On-Time Rate" variant="green" icon={<Clock className="h-4 w-4" />} />
      <StatRing value={91} displayValue="91" label="Safety Score" variant="purple" icon={<ShieldCheck className="h-4 w-4" />} />
      <StatRing value={20} displayValue="2" label="Incidents" variant="orange" icon={<AlertTriangle className="h-4 w-4" />} />
    </div>
  ),
};
